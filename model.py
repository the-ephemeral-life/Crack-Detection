import os
from pathlib import Path
from ultralytics import YOLO

def main():
    """
    Main function to run the entire YOLOv8 OBB training, validation, and testing pipeline.
    """
    # --- 1. SETUP ---
    # Define the root directory of your project (the 'dp' folder)
    # This makes all other paths relative to the project root, which is robust.
    ROOT_DIR = Path(__file__).resolve().parents[1]
    
    # Define the path to your data.yaml file
    DATA_YAML_PATH = ROOT_DIR / 'data' / 'data.yaml'
    
    # Define the pre-trained model we'll use for transfer learning
    PRE_TRAINED_MODEL = 'yolov8n-obb.pt'
    
    # Define training parameters
    EPOCHS = 50
    IMG_SIZE = 640

    RUN_NAME = 'crack_detector_run_1'
    
    print(f"Project Root Directory: {ROOT_DIR}")
    print(f"Data YAML Path: {DATA_YAML_PATH}")
    print("-" * 30)

    print("🚀 Starting model training...")
    
    # Load the pre-trained OBB model
    model = YOLO(PRE_TRAINED_MODEL)
    
    # Train the model using your dataset
    # The results object contains all information about the training run.
    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project='runs',  # All runs will be saved in the 'dp/runs' directory
        name=RUN_NAME,
        exist_ok=True # Allows re-running the script without error
    )
    
    print("✅ Training complete.")
    print(f"Best model saved at: {results.save_dir / 'weights' / 'best.pt'}")
    print("-" * 30)

    # --- 3. VALIDATION ---
    print("🧪 Starting model validation...")
    
    # Load the best performing model from the training run
    best_model_path = results.save_dir / 'weights' / 'best.pt'
    best_model = YOLO(best_model_path)
    
    # Run validation. The metrics object contains performance data.
    metrics = best_model.val()
    
    print("✅ Validation complete.")
    print("Validation Metrics:")
    print(f"  mAP50-95 (Box): {metrics.box.map:.4f}")
    print(f"  mAP50 (Box):   {metrics.box.map50:.4f}")
    print(f"  mAP75 (Box):   {metrics.box.map75:.4f}")
    print("-" * 30)

    # --- 4. TESTING (INFERENCE) ---
    print("🔎 Starting testing on new images...")
    
    # Define the directory with test images
    test_images_dir = ROOT_DIR / 'data' / 'test' / 'images'
    
    if not test_images_dir.is_dir():
        print(f"⚠️ Test directory not found at: {test_images_dir}")
        print("Skipping testing.")
    else:
        # Run predictions on the test images
        # The results are saved automatically to 'dp/runs/crack_detector_run_1/...'
        test_results = best_model.predict(
            source=str(test_images_dir),
            save=True  # Saves the images with bounding boxes drawn on them
        )
        print("✅ Testing complete.")
        # The save directory is part of the first result object
        print(f"Test prediction images saved in: {test_results[0].save_dir}")
    
    print("-" * 30)
    
    # --- 5. ANALYTICS AND ERRORS ---
    print("📊 All analytics and results are saved in the run directory:")
    print(f"   {results.save_dir}")
    print("\nInside this folder, you will find:")
    print("  - `weights/best.pt`: Your best trained model file for future use.")
    print("  - `results.png`: A chart showing training and validation loss, mAP, and other metrics over epochs.")
    print("  - `confusion_matrix.png`: A matrix showing any prediction errors between classes.")
    print("  - `val_batch*_pred.jpg`: Images from the validation set with your model's predictions drawn on them.")
    print("  - And much more for detailed analysis!")
    print("\n🎉 Pipeline finished successfully!")


if __name__ == '__main__':
    main()