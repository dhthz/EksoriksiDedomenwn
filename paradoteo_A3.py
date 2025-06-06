import sys
import os
from helper_python_files.binary_model import binary_nn
from helper_python_files.multiclass_nn import multiclass_nn
from helper_python_files.svm import run_svm

# Add the current directory to the path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the models


def main():
    print("Running Binary Neural Network...")
    binary_nn()

    print("\nRunning Multiclass Neural Network...")
    multiclass_nn()

    print("\nRunning Support Vector Machine...")
    run_svm()

    print("\nAll models completed execution.")


if __name__ == "__main__":
    main()
