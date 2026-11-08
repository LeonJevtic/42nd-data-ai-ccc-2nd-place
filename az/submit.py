"""Copy a trained model to submit/model.onnx and run the official validator on it.

python -m az.submit runs/r1/gen5.onnx
"""
import os
import shutil
import subprocess
import sys


def main(src):
    os.makedirs("submit", exist_ok=True)
    dst = os.path.join("submit", "model.onnx")
    shutil.copy(src, dst)
    with open(os.path.join("submit", "SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write(os.path.abspath(src) + "\n")
    r = subprocess.run([sys.executable, "tools/validate_submission.py", dst], capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(f"size {os.path.getsize(dst) / 1e6:.2f} MB -> {os.path.abspath(dst)}")


if __name__ == "__main__":
    main(sys.argv[1])
