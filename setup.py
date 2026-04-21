"""
One-shot setup script.
Runs the dataset generator and then trains the classifier.
Equivalent to:
    python src/generate_dataset.py
    python src/train_model.py
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=HERE)


if __name__ == "__main__":
    py = sys.executable
    run([py, os.path.join("src", "generate_dataset.py")])
    run([py, os.path.join("src", "train_model.py")])
    print("\nSetup complete.")
    print("Run the web app with:  python app.py")
    print("Then open             http://localhost:5000")
