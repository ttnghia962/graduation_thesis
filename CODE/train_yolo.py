# train_yolo.py

from ultralytics import YOLO
import yaml
from pathlib import Path

def train_yolo_model(data_yaml: str, epochs: int = 100):
    """
    Train YOLO model for flowchart shape detection
    """
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        patience=50,
        device='0',  # Use GPU if available
        project='flowchart_detection',
        name='exp1'
    )
    
    return results

def setup_training(data_dir: str = "dataset"):
    """
    Setup and start training
    """
    data_dir = Path(data_dir)
    data_yaml = str(data_dir / "data.yaml")
    
    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    # Load and verify data.yaml
    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)
        
    print("Training Configuration:")
    print(f"Classes: {data_config['names']}")
    print(f"Number of classes: {data_config['nc']}")
    
    # Start training
    results = train_yolo_model(data_yaml)
    return results

if __name__ == "__main__":
    setup_training()