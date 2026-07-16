# מדריך למבנה הפרויקט ול־Workflow – חומר למצגת

מסמך זה מסביר את מבנה הפרויקט לפי המימוש הנוכחי בריפו והעדכונים האחרונים. הוא מתאר את אבני הבניין, קובצי ההרצה, זרימת המידע והתוצרים, וניתן להשתמש בו כבסיס לכתיבת המצגת.

## תמונת־על של ה־Workflow

```text
COCO128 clean images
        |
        v
distortions.py
יצירת 4 עיוותים × 4 רמות
        |
        v
run_30_pic_dataset.py
הרצת 4 משימות על clean ועל distorted
        |
        +-- Classical: Template Matching, Optical Flow
        +-- Deep Learning: Object Detection, Instance Segmentation
        |
        v
metadata_summary_base.csv
+ distorted images
+ annotated task images
        |
        v
apply_enhancements.py
הפעלת enhancement מתאים לכל עיוות
        |
        v
enhanced_images/
        |
        v
evaluate_enhancements.py
הרצה חוזרת של אותן 4 משימות
        |
        v
שורות Enhanced ב־metadata_summary_base.csv
        |
        v
plot_enhancement_results.py
גרפי Baseline מול Enhanced
```

בנפרד קיים מסלול Fine-Tuning:

```text
best.pt
   |
   v
evaluate_finetuned.py
   +-- detected_objects
   +-- avg_confidence
   |
   v
שורות Fine-Tuned ב־metadata_summary_base.csv
   |
   v
plot_finetune_results.py
```

הקוד של מסלול ה־Fine-Tuned תוקן, אך אין להריץ אותו לפני ש־`best.pt` זמין והנתונים הקיימים גובו וטופלו.

## 1. `distortions.py` – יצירת תנאי הניסוי

הקובץ מכיל פונקציות עיבוד תמונה טהורות. הוא אינו מריץ מודלים ואינו שומר קבצים בעצמו. כל פונקציה מקבלת תמונת OpenCV בפורמט BGR מסוג `uint8` ומחזירה תמונה מעוותת.

### חישוב SNR

`calculate_snr()` משווה בין תמונת המקור לתמונה המעוותת ומחשב Signal-to-Noise Ratio בדציבלים:

$$SNR=10\log_{10}\left(\frac{signal\ power}{noise\ power}\right)$$

עוצמת האות מחושבת מסכום ריבועי הפיקסלים בתמונה הנקייה, ועוצמת הרעש מסכום ריבועי ההפרשים. אם התמונות זהות מוחזר infinity. SNR מודד את עוצמת השינוי בתמונה, ולא את הצלחת משימת הראייה.

### Gaussian noise

נוסף רעש מהתפלגות נורמלית עם ממוצע אפס. רמות ה־sigma הן 15, 30, 50 ו־75. לאחר הוספת הרעש הפיקסלים נחתכים לטווח 0–255.

### Salt & pepper

פיקסלים אקראיים הופכים לשחור או לבן. צפיפות הפיקסלים הפגועים היא 2%, 5%, 15% ו־30%. מחצית מהפיקסלים הנבחרים הם salt ומחצית pepper.

### Low light

התאורה החלשה מיוצרת באמצעות שילוב של הקטנת brightness ושינוי gamma. רמות ה־scale הן 0.70, 0.50, 0.30 ו־0.15, ורמות gamma הן 0.80, 0.60, 0.40 ו־0.30.

### Motion blur

נבנה kernel אלכסוני המדמה תנועה בכיוון של כ־45 מעלות. גדלי ה־kernel הם 5×5, 11×11, 21×21 ו־35×35. ככל שה־kernel ארוך יותר, המריחה חזקה יותר.

ניסוח אפשרי למצגת: יצרנו ארבע משפחות של degradations, כל אחת בארבע רמות חומרה. העיוותים אינם משנים את הגאומטריה, ולכן תוויות ה־bounding boxes המקוריות נשארות תקפות. לכל תמונה חישבנו גם SNR כדי לכמת את עוצמת הפגיעה.

## 2. `classical_tasks.py` – המשימות הקלאסיות

המחלקה `ClassicalTasks` מכילה את האלגוריתמים עצמם ואינה עוסקת ב־CSV או בתיקיות.

### Template Matching

המערכת מחפשת template קטן בתוך תמונה גדולה באמצעות `cv2.matchTemplate()` עם `TM_CCOEFF_NORMED`. הפונקציה מחזירה מפת התאמה, את המיקום הטוב ביותר ואת ציון ההתאמה. ציון קרוב ל־1 מעיד בדרך כלל על התאמה חזקה.

בניסוי של 30 התמונות, ה־template הוא crop בגודל של עד 100×100 פיקסלים ממרכז תמונת המקור. אותו template נקי משמש לחיפוש בתוך clean ובתוך distorted, כדי למדוד כיצד העיוות פוגע בהתאמה.

### Optical Flow

זהו Sparse Optical Flow. תחילה ממירים את התמונות ל־grayscale, מזהים עד 200 נקודות Shi–Tomasi באמצעות `goodFeaturesToTrack()`, ולאחר מכן עוקבים אחריהן באמצעות pyramidal Lucas–Kanade ו־`calcOpticalFlowPyrLK()`.

המדד הוא `tracked_points`: מספר הנקודות שסומנו כנקודות שנעקבו בהצלחה. זהו מדד פעילות ולא Ground-Truth accuracy; רעש יכול ליצור פינות מלאכותיות ולהגדיל את מספר הנקודות גם כאשר המעקב אינו איכותי יותר.

## 3. `yolo_tasks.py` – עטיפה למודלי YOLO

המחלקה `YoloTasks` מרכזת את טעינת המודלים והקריאות הבסיסיות ל־YOLO:

- `yolov8n.pt` עבור Object Detection.
- `yolov8n-seg.pt` עבור Instance Segmentation.

`detect_objects()` ו־`segment_instances()` מריצות prediction עם גודל תמונה 320 וסף confidence של 0.25. התוצאות כוללות boxes, classes, confidence, ובמקרה של segmentation גם masks.

`evaluate_baseline_mAP()` מפעילה `model.val()` מול Dataset YAML. זהו מסלול שונה מ־prediction: `predict()` מחזיר זיהויים, ואילו `val()` משווה ל־Ground Truth ומחשב mAP.

## 4. `run_classical_experiments.py`

לקובץ שני תפקידים.

### פונקציות Evaluation לשימוש חוזר

`evaluate_optical_flow()` קוראת לאלגוריתם, סופרת נקודות ועורכת visualization עם חצים ירוקים. היא מחזירה מבנה אחיד הכולל `metrics` ו־`visualized_image`.

`evaluate_template_matching()` מחשבת את ההתאמה, מציירת מלבן סביב המיקום הטוב ביותר ומחזירה `matching_score`, `location` ותמונה מסומנת.

המבנה המשותף מאפשר ל־runner הראשי לטפל במשימות הקלאסיות ובמשימות ה־DL באופן אחיד.

### Runner ישן על תמונה אחת

`ClassicalExperimentRunner` הוא שלב מוקדם של הפרויקט. הוא משתמש בתמונה `000000000009.jpg`, יוצר עיוותים, שומר before/after, CSV וגרפים נפרדים. המתודות מסומנות כ־legacy/deprecated. הניסוי המרכזי עבר ל־`run_30_pic_dataset.py`.

ניסוח אפשרי למצגת: בתחילת העבודה כל משימה נבדקה בנפרד על תמונה מייצגת. בהמשך הפרדנו את חישוב המשימה מפעולות הקלט והפלט, וכך יכולנו להרחיב את הניסוי ל־30 תמונות.

## 5. `run_dl_experiments.py`

גם כאן יש פונקציות פעילות ו־runner ישן.

### Object Detection

`evaluate_object_detection()` מקבלת תמונת RGB ומודל YOLO, מריצה prediction בגודל 320, סופרת boxes ומחשבת confidence ממוצע. הפלט כולל:

- `detected_objects` – מספר הזיהויים.
- `avg_confidence` – ממוצע הביטחון בזיהויים.
- `visualized_image` – תמונה עם bounding boxes.

### Instance Segmentation

`evaluate_segment_instances()` משתמשת ב־YOLO segmentation ומחזירה:

- `segmented_instances` – מספר המופעים.
- `avg_confidence` – confidence ממוצע.
- תמונה עם masks ו־boxes.

המדדים הללו הם Activity Metrics ולא accuracy: מספר זיהויים או confidence גבוה אינם מבטיחים שהזיהויים נכונים. לצורך accuracy מול Ground Truth משתמשים ב־mAP.

`DeepLearningExperimentRunner` הוא runner מוקדם על תמונה אחת. הוא אינו ה־pipeline המרכזי. הביטוי הישן `mAP Proxy` אינו מדויק: confidence אינו mAP.

## 6. `run_30_pic_dataset.py` – ה־Runner המרכזי

זהו ה־orchestrator המרכזי של שלב ה־Baseline וה־Distortions. הוא אינו מממש מחדש את האלגוריתמים, אלא מחבר בין פונקציות העיוות, ארבע המשימות, שמירת הקבצים וה־CSV.

### אתחול ובחירת תמונות

ה־runner מוגדר עם 30 תמונות, ארבע משימות, ארבעה עיוותים וארבע רמות. שני מודלי YOLO נטענים פעם אחת. שמות התמונות ממוינים ונבחרות 30 התמונות הראשונות, ולכן המדגם דטרמיניסטי לפי שמות ולא אקראי.

### הכנת inputs קלאסיים

לכל תמונה נוצר template ממרכז התמונה ו־frame שני סינתטי באמצעות motion blur ברמה 3. זה מאפשר להפעיל optical flow גם כאשר COCO הוא Dataset של תמונות סטטיות ולא של וידאו.

### Clean Baseline

לכל תמונת מקור מופעלות ארבע המשימות על המצב הנקי. נשמרות visualizations והמדדים נרשמים עם `distortion_type=clean`, `level=0` ו־SNR אינסופי.

### Distorted Conditions

לכל אחת מ־30 התמונות מופעלים ארבעה עיוותים בארבע רמות, ולכן נוצרים 480 קובצי תמונה מעוותים. לכל condition מחושב SNR ומופעלות ארבע המשימות.

עבור Optical Flow, גם frame 1 וגם frame 2 עוברים את אותו distortion ובאותו level.

### טיפול בשגיאות

כל משימה עטופה ב־`try/except`. אם משימה נכשלת, ה־pipeline ממשיך ושומר ערכי fallback, בדרך כלל אפס ותמונה לא מסומנת. כך כישלון מקומי אינו מפיל ניסוי ארוך.

### תוצרי התמונות

התמונות המעוותות נשמרות ב־`data/distorted_images/{distortion}_l{level}/`.

התמונות המסומנות נשמרות ב־`data/tasks_applied_on_distorted/{task}/{condition}/`. Detection כולל boxes, segmentation כולל masks, template matching כולל מלבן, ו־optical flow כולל חצים.

### מבנה ה־CSV

כל metric נשמר בשורה נפרדת עם שם התמונה, סוג העיוות, level, SNR, שם המשימה, שם המדד, ערך המדד ונתיבי הקבצים. בהמשך נוספה עמודת `model_type` עם Baseline, Enhanced או Fine-Tuned.

לכל condition יש בפועל שבע שורות metrics: אחת ל־Optical Flow, שתיים ל־Template Matching, שתיים ל־Segmentation ושתיים ל־Detection. לכן עבור Baseline קיימות 3,570 שורות. המספר 2,040 שמופיע ב־docstring הישן אינו נכון לאחר flattening של כל המדדים.

## 7. `generate_plots.py` – גרפי Degradation

הקובץ קורא את `metadata_summary_base.csv` ומייצר לכל משימה גרף מול level וגרף מול SNR:

- `{task}_vs_level.png`
- `{task}_vs_snr.png`

הגרפים מתארים כיצד מדד הפעילות משתנה ככל שהעיוות חזק יותר.

## 8. `enhancements.py` – שיטות השיקום

הקובץ מכיל פונקציות enhancement טהורות שאינן שומרות קבצים ואינן מריצות משימות.

### Gaussian filter

`GaussianBlur` עם kernel בגודל 5×5 נבחר לטיפול ב־Gaussian noise. הוא מפחית רעש רציף במחיר של איבוד פרטים מסוים.

### Median filter

`medianBlur` עם kernel בגודל 5×5 נבחר ל־salt & pepper. החציון מסיר פיקסלים חריגים בצורה יעילה יחסית ושומר טוב יותר על קצוות.

### CLAHE

התמונה מומרת ל־LAB, CLAHE מופעל רק על ערוץ הבהירות L, ולאחר מכן הערוצים מחוברים מחדש. הפרמטרים הם `clipLimit=2.0` ו־tile grid של 8×8. השיטה משפרת contrast מקומי ומתאימה ל־low light.

### Sharpening

מופעל kernel שמדגיש קצוות. הוא נבחר כניסיון לטפל ב־motion blur, אבל אינו deblurring אמיתי. התוצאות הראו שבמרבית הרמות הוא אף פגע ב־mAP, מפני שהוא מדגיש מריחה קיימת במקום לשחזר מידע שאבד.

### מפת ההתאמה

`ENHANCEMENT_FOR_DISTORTION` היא מקור האמת היחיד:

| Distortion | Enhancement | רציונל |
|---|---|---|
| Gaussian noise | Gaussian filter | הפחתת רעש רציף |
| Salt & pepper | Median filter | הסרת פיקסלים חריגים |
| Low light | CLAHE | שיפור contrast מקומי |
| Motion blur | Sharpening | הדגשת קצוות; לא deblurring אמיתי |

## 9. `apply_enhancements.py` – יצירת התמונות המשופרות

הסקריפט קורא את טבלת המטא־דאטה, שולף שילובים ייחודיים של תמונה, distortion ו־level ומסיר כפילויות שנובעות מכך שלכל תמונה יש מספר metrics.

לכל תמונה הוא טוען את `distorted_image_path`, בוחר את ההשבחה באמצעות `ENHANCEMENT_FOR_DISTORTION` ושומר את התוצאה תחת `data/enhanced_images/{distortion}_l{level}/`.

שלב זה רק יוצר 480 תמונות enhanced. הוא אינו מריץ YOLO ואינו משנה את ה־CSV. ההפרדה מאפשרת לבדוק את התמונות לפני הערכת המשימות.

## 10. `evaluate_enhancements.py` – הערכת התמונות המשופרות

הסקריפט מפעיל מחדש את אותן ארבע משימות על התמונות המשופרות. הוא טוען את מודלי YOLO פעם אחת, משחזר את ה־template ואת frame 2, ושומר visualizations תחת `data/tasks_applied_on_enhanced/`.

עבור Optical Flow נדרש טיפול מיוחד: frame 2 לא נשמר בשלב הראשון, ולכן הוא נוצר מחדש, עובר את אותו distortion ואת אותו enhancement, ורק אז מופעל flow בין שני frames שעברו את אותו מסלול.

לכל metric נוצרת שורה חדשה עם `model_type=Enhanced`. ב־CSV הנוכחי יש 3,360 שורות Enhanced, כלומר 480 conditions כפול שבעה metrics.

## 11. `plot_enhancement_results.py` – גרפי Recovery

לכל משימה נבחר Primary Metric אחד:

- Object Detection: `detected_objects`.
- Instance Segmentation: `segmented_instances`.
- Template Matching: `matching_score`.
- Optical Flow: `tracked_points`.

כך לא מערבבים metrics שונים כמו detection count ו־confidence, ולא מנסים לשרטט שדות לא מספריים כמו location.

הסקריפט משווה רק בין שורות Baseline ו־Enhanced ומסיר clean level 0, מפני שאין enhancement לתמונה הנקייה.

לכל משימה נוצרים שני גרפים:

1. Bar chart המסכם את הממוצע לפי distortion על פני כל הרמות.
2. Line chart המציג Baseline מול Enhanced בכל level, עם panel נפרד לכל distortion.

בסך הכול נוצרים שמונה גרפים תחת `data/tasks_graphs_and_tables/plots/`.

## 12. מסלול Fine-Tuned לאחר התיקון

`evaluate_finetuned.py` אמור לאתר את `best.pt`, לבחור condition ייחודי לפי image name, distortion ו־level, ולהריץ inference פעם אחת בלבד.

הבאג המקורי היה שהסקריפט עבר גם על שורת `detected_objects` וגם על שורת `avg_confidence`, הריץ את אותה תמונה פעמיים וכתב בשתיהן את מספר הזיהויים.

לאחר התיקון, inference יחיד יוצר במפורש:

- `detected_objects` – מספר ה־boxes.
- `avg_confidence` – ממוצע confidence של ה־boxes, או 0 אם אין זיהויים.

נוספו בדיקות ל־metrics חסרים, כפילויות, confidence מחוץ ל־0–1 ו־count שאינו מספר שלם ולא שלילי.

`plot_finetune_results.py` מפריד כעת בין הגרפים:

- `finetune_recovery_bar.png` ו־`finetune_recovery_lines.png` משתמשים רק ב־detection count.
- `finetune_confidence_bar.png` ו־`finetune_confidence_lines.png` משתמשים רק ב־confidence, עם ציר 0–1.

הקוד מוכן, אבל הגרפים החדשים טרם נוצרו משום שההערכה לא הורצה ללא `best.pt`.

## 13. טבלת התוצרים המרכזיים

| שלב | קובץ מרכזי | קלט | פלט |
|---|---|---|---|
| יצירת עיוות | `distortions.py` | תמונה נקייה ו־level | תמונה מעוותת |
| משימות קלאסיות | `classical_tasks.py` | תמונה/template או frames | תוצאות matching/flow |
| Evaluation קלאסי | `run_classical_experiments.py` | inputs למשימה | metrics ותמונה מסומנת |
| משימות DL | `run_dl_experiments.py` | תמונת RGB ומודל | counts, confidence ותמונה מסומנת |
| ניסוי 30 תמונות | `run_30_pic_dataset.py` | 30 תמונות נקיות | 480 distorted, visualizations ו־Baseline CSV |
| הגדרת enhancement | `enhancements.py` | תמונה מעוותת | תמונה משופרת |
| יצירת enhanced | `apply_enhancements.py` | distorted images | 480 enhanced images |
| הערכת enhanced | `evaluate_enhancements.py` | enhanced images | visualizations ושורות Enhanced |
| גרפי enhanced | `plot_enhancement_results.py` | Baseline/Enhanced rows | שמונה recovery plots |
| הערכת Fine-Tuned | `evaluate_finetuned.py` | `best.pt` ו־conditions | Fine-Tuned count/confidence rows |
| גרפי Fine-Tuned | `plot_finetune_results.py` | Fine-Tuned rows | count ו־confidence plots |
| GT box-mAP | `evaluate_map_gt.py` | images ותוויות COCO | `map_summary.csv` |

## 14. חלוקה אפשרית לשקופיות

### שקופית 1 – Experiment Pipeline

Clean → Distortion → Task Evaluation → Enhancement → Re-evaluation. לציין ארבעה distortions, ארבע רמות וארבע משימות.

### שקופית 2 – Distortions

Gaussian noise, salt & pepper, low light ו־motion blur, כל אחד בארבע רמות. SNR משמש לכימות עוצמת הפגיעה.

### שקופית 3 – Vision Tasks

Classical: Template Matching ו־Sparse Optical Flow. Deep Learning: YOLOv8n Object Detection ו־YOLOv8n-seg Instance Segmentation.

### שקופית 4 – Experiment Runner

ה־runner בוחר 30 תמונות, יוצר 480 גרסאות מעוותות, מפעיל ארבע משימות ושומר metrics ו־annotated images בטבלה אחידה.

### שקופית 5 – Enhancement Methods

להציג את ההתאמה Gaussian–Gaussian filter, salt & pepper–median, low light–CLAHE, motion blur–sharpening.

### שקופית 6 – Enhancement Evaluation

אותן תמונות, אותן משימות ואותם metrics. Bar plots מסכמים, ו־line plots מראים את השינוי לפי severity.

### שקופית 7 – Fine-Tuning

מודל pretrained עובר fine-tuning ונשמר כ־`best.pt`. Evaluation מחזיר detection count ו־confidence בנפרד. Box-mAP מול GT נדרש לקביעת accuracy אמיתי.

### שקופית 8 – Activity מול Accuracy

| Activity Metric | GT Accuracy |
|---|---|
| Number of detections | Box-mAP |
| Average confidence | Precision, Recall, AP |
| Number of segments | Mask-mAP |
| Tracked points | דורש GT ייעודי שאינו קיים כאן |

המסר המרכזי: מדדי activity מתארים כמה האלגוריתם עשה; mAP בודק אם הוא צדק.

## 15. נוסח קצר להצגה בעל פה

בנינו pipeline מודולרי שבו פונקציות העיוות, ההשבחה והמשימות עצמן מופרדות מקוד ההרצה. תחילה מימשנו שתי משימות קלאסיות ושתי משימות מבוססות YOLO. לאחר בדיקות ראשוניות על תמונה יחידה, הרחבנו את הניסוי ל־30 תמונות מ־COCO128. לכל תמונה יצרנו ארבעה סוגי עיוות בארבע רמות חומרה, כלומר 480 תמונות מעוותות, והרצנו עליהן את כל ארבע המשימות. לכל הרצה שמרנו גם מדד מספרי וגם visualization, וריכזנו את הנתונים בטבלת CSV אחידה. בשלב השיקום התאמנו לכל עיוות enhancement קלאסי, יצרנו גרסאות משופרות והרצנו עליהן מחדש בדיוק את אותן משימות. כך יכולנו להשוות בין המצב המעוות למצב המשופר לפי סוג העיוות ורמת החומרה. בנוסף קיים מסלול fine-tuning של YOLO, שבו מודל `best.pt` נבדק לפי מספר זיהויים ו־confidence, ובהמשך יש להשלים את ההשוואה המרכזית באמצעות box-mAP מול Ground Truth.

