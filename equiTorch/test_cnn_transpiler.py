import os
from pathlib import Path

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

from transpiler import PyTorchToEquinoxTranspiler

# Read the PyTorch CNN example code with the correct path
pytorch_code = Path(os.path.join(script_dir, "pytorch_cnn_example.py")).read_text()

transpiler = PyTorchToEquinoxTranspiler()

equinox_code = transpiler.transpile(pytorch_code)

print("Transpiled Equinox code:")
print("-" * 50)
print(equinox_code)
print("-" * 50)

# Save to a file in the same directory as this script
output_path = os.path.join(script_dir, "equinox_cnn.py")
with open(output_path, "w") as f:
    f.write(equinox_code)

print(f"Saved transpiled code to '{output_path}'")
