from pathlib import Path

# Import the transpiler
from transpiler import PyTorchToEquinoxTranspiler

# Read the PyTorch CNN example code
pytorch_code = Path("pytorch_cnn_example.py").read_text()

# Initialize the transpiler
transpiler = PyTorchToEquinoxTranspiler()

# Transpile the code
equinox_code = transpiler.transpile(pytorch_code)

# Print the transpiled code
print("Transpiled Equinox code:")
print("-" * 50)
print(equinox_code)
print("-" * 50)

# Optionally save to a file
with open("equinox_cnn.py", "w") as f:
    f.write(equinox_code)

print("Saved transpiled code to 'equinox_cnn.py'")
