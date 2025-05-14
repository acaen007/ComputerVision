"""
Main pipeline script to select and run inference for DETR, RESNET, or VGG16 models.
"""
import sys
import os

# Define the project root directory dynamically
# This assumes main.py is at the root of the project_pipeline directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# Add project root to sys.path to allow for absolute imports from model directories
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the refactored inference functions
# These imports will only succeed if the respective __init__.py files are present
# or if the structure allows direct module import (which it should with PROJECT_ROOT in sys.path)
try:
    from ANDY.inference.andy_inference import run_detr_inference
except ImportError as e:
    print(f"Could not import DETR inference function: {e}")
    print("Ensure ANDY/inference/andy_inference.py exists and is structured correctly.")
    run_detr_inference = None

try:
    from BERNARDO.inference import run_resnet_inference
except ImportError as e:
    print(f"Could not import RESNET inference function: {e}")
    print("Ensure BERNARDO/inference/inference.py exists and is structured correctly.")
    run_resnet_inference = None

try:
    from GONCALO.inference.inference_pipeline import run_vgg16_inference
except ImportError as e:
    print(f"Could not import VGG16 inference function: {e}")
    print("Ensure GONCALO/inference/inference_pipeline.py exists and is structured correctly.")
    run_vgg16_inference = None

def create_output_directories(model_name):
    """Creates the main output directory and a model-specific subdirectory."""
    main_output_dir = os.path.join(PROJECT_ROOT, "output")
    model_output_dir = os.path.join(main_output_dir, model_name.upper())
    try:
        os.makedirs(main_output_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)
        print(f"Output directory created/ensured: {model_output_dir}")
    except OSError as e:
        print(f"Error creating output directories: {e}")
        return False
    return True

def main_pipeline():
    """Main function to drive the model selection and inference pipeline."""
    print("Welcome to the Model Inference Pipeline!")
    print(f"Project Root: {PROJECT_ROOT}")

    while True:
        model_choice = input("Which model would you like to run? (DETR, RESNET, VGG16, or type 'exit' to quit): ").strip().upper()

        if model_choice == "EXIT":
            print("Exiting pipeline.")
            break

        if model_choice not in ["DETR", "RESNET", "VGG16"]:
            print("Invalid model choice. Please enter DETR, RESNET, or VGG16.")
            continue

        if not create_output_directories(model_choice):
            print(f"Could not create output directories for {model_choice}. Aborting inference.")
            continue

        print(f"\nAttempting to run {model_choice} model...")

        if model_choice == "DETR":
            if run_detr_inference:
                try:
                    run_detr_inference()
                except Exception as e:
                    print(f"An error occurred during DETR inference: {e}")
            else:
                print("DETR inference function is not available due to import errors.")
        
        elif model_choice == "RESNET":
            if run_resnet_inference:
                try:
                    run_resnet_inference(PROJECT_ROOT)
                except Exception as e:
                    print(f"An error occurred during RESNET inference: {e}")
            else:
                print("RESNET inference function is not available due to import errors.")

        elif model_choice == "VGG16":
            if run_vgg16_inference:
                try:
                    run_vgg16_inference(PROJECT_ROOT)
                except Exception as e:
                    print(f"An error occurred during VGG16 inference: {e}")
            else:
                print("VGG16 inference function is not available due to import errors.")
        
        print(f"\n{model_choice} model run attempt finished.")
        print("Check the 'output' directory for results or error messages.")
        print("-----------------------------------------------------")

if __name__ == "__main__":
    # Create dummy __init__.py files if they don't exist to help with imports
    # This is a helper for the sandbox environment and might not be needed in a user's setup
    # if their Python path is configured correctly or they use a proper package structure.
    for component_dir in ["ANDY", "BERNARDO", "GONCALO"]:
        for sub_dir in ["inference", "models", "utils"]:
            init_path = os.path.join(PROJECT_ROOT, component_dir, sub_dir, "__init__.py")
            if not os.path.exists(init_path):
                # Create empty __init__.py if the directory exists
                if os.path.isdir(os.path.dirname(init_path)):
                    try:
                        with open(init_path, "w") as f:
                            pass # Create empty file
                    except Exception as e:
                        print(f"Warning: Could not create dummy __init__.py at {init_path}: {e}")
    
    main_pipeline()

