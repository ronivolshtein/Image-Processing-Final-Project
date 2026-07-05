from ultralytics import YOLO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Load the pre-trained baseline model (using the nano version as a default)
    # If you used a different size in Week 1 (like yolov8s.pt), change it here.
    model = YOLO(str(PROJECT_ROOT / 'yolov8n.pt'))

    yaml_path = (PROJECT_ROOT / 'data' / 'yolo_finetune' / 'finetune_dataset.yaml').resolve()
    output_project = PROJECT_ROOT / 'runs' / 'detect'

    print("🚀 Starting YOLO Fine-Tuning on distorted images...")

    # Train the model
    model.train(
        data=str(yaml_path),
        epochs=10,         # A short run for fine-tuning
        imgsz=640,         # Standard YOLO image size
        batch=8,           # A safe batch size for local machines
        project=str(output_project),
        name='finetune_distorted',
        exist_ok=True      # Overwrite if we run the exact same experiment again
    )

    print("✅ Fine-Tuning completed successfully!")
    print(f"📂 Results, weights, and graphs are saved to: {output_project / 'finetune_distorted'}")

if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    from multiprocessing import freeze_support
    freeze_support()
    main()