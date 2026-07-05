from ultralytics import YOLO
from pathlib import Path

def main():
    # Load the pre-trained baseline model (using the nano version as a default)
    # If you used a different size in Week 1 (like yolov8s.pt), change it here.
    model = YOLO('yolov8n.pt') 

    yaml_path = Path('data/yolo_finetune/finetune_dataset.yaml').absolute()

    print("🚀 Starting YOLO Fine-Tuning on distorted images...")
    
    # Train the model
    results = model.train(
        data=str(yaml_path),
        epochs=10,         # A short run for fine-tuning
        imgsz=640,         # Standard YOLO image size
        batch=8,           # A safe batch size for local machines
        project='runs/detect',
        name='finetune_distorted',
        exist_ok=True      # Overwrite if we run the exact same experiment again
    )
    
    print("✅ Fine-Tuning completed successfully!")
    print("📂 Results, weights, and graphs are saved to: runs/detect/finetune_distorted")

if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    from multiprocessing import freeze_support
    freeze_support()
    main()