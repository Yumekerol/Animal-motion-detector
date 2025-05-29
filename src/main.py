from detector.core import RoboflowAnimalDetector
import os
def main():
    detector = RoboflowAnimalDetector()

    print("🐾 Enhanced Animal Detector with Multiple Datasets")
    print("=" * 60)
    print("1. Download and merge multiple datasets")
    print("2. Validate merged dataset")
    print("3. Train model (Nano - Fast)")
    print("4. Train model (Small - Balanced)")
    print("5. Train model (Medium - Accurate)")
    print("6. Train model (Large - Very Accurate)")
    print("7. Test model (Webcam)")
    print("8. Test model (Image file)")
    print("9. Test model (Video file)")
    print("10. Evaluate model performance")
    print("11. Run full pipeline")
    print("=" * 60)

    choice = input("Enter your choice (1-11): ").strip()

    if choice == "1":
        detector.download_and_merge_datasets()

    elif choice == "2":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.validate_dataset()

    elif choice == "3":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=100, model_size='n')

    elif choice == "4":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=200, model_size='s')

    elif choice == "5":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=300, model_size='m')

    elif choice == "6":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.train_model(epochs=300, model_size='l')

    elif choice == "7":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_webcam()

    elif choice == "8":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_on_image()

    elif choice == "9":
        if detector.class_names or (detector.load_dataset_config()):
            detector.test_model_on_video()

    elif choice == "10":
        if os.path.exists(detector.merged_dataset_path) or detector.download_and_merge_datasets():
            detector.load_dataset_config()
            detector.evaluate_model()

    elif choice == "11":
        print("🚀 Running full pipeline...")

        print("\n📥 Step 1: Download and merge datasets")
        if detector.download_and_merge_datasets():
            detector.load_dataset_config()

            print("\n🔍 Step 2: Validate merged dataset")
            if detector.validate_dataset():

                print("\n🚀 Step 3: Train model")
                if detector.train_model(epochs=100, model_size='s'):
                    print("\n📊 Step 4: Evaluate model")
                    detector.evaluate_model()

                    print("\n🎥 Step 5: Test model")
                    test_choice = input("Test with (w)ebcam, (i)mage, or (v)ideo? ").strip().lower()
                    if test_choice == 'w':
                        detector.test_model_webcam()
                    elif test_choice == 'i':
                        detector.test_model_on_image()
                    elif test_choice == 'v':
                        detector.test_model_on_video()

    else:
        print("❌ Invalid option")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install ultralytics opencv-python roboflow pyyaml")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")