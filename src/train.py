import os
import sys
from pathlib import Path
import importlib

def run_centralized_training():
    print("Starting centralized training for all blends...")

    # Define the project root and adjust sys.path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))

    # List of blend training modules to run
    blend_trainers = {
        1: 'src.training.blend1_train',
        2: 'src.training.blend2_train',
        3: 'src.training.blend3_train',
        4: 'src.training.blend4_train',
        5: 'src.training.blend5_train',
        6: 'src.training.blend6_train',
        7: 'src.training.blend7_train',
        8: 'src.training.blend8_train',
        9: 'src.training.blend9_train',
        10: 'src.training.blend10_train'
    }

    for blend, module_path in blend_trainers.items():
        print(f"\nRunning training for BlendProperty{blend}...")
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, f'Blend{blend}Trainer'):
                trainer_class = getattr(module, f'Blend{blend}Trainer')
                trainer_class.train()
            else:
                print(f"Error: No trainer class found for BlendProperty{blend} in {module_path}")
        except Exception as e:
            print(f"Error training BlendProperty{blend}: {str(e)}")

if __name__ == "__main__":
    run_centralized_training()