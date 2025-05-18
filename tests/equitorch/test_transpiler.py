from pathlib import Path
from transpiler import PyTorchToEquinoxTranspiler

# Read the PyTorch MLP example code
pytorch_code = Path("pytorch_mlp_example.py").read_text()

# Create a transpiler
transpiler = PyTorchToEquinoxTranspiler()

# Transpile the PyTorch code to Equinox code
equinox_code = transpiler.transpile(pytorch_code)

print("Transpiled Equinox code using improved transpiler:")
print("-" * 50)
print(equinox_code)
print("-" * 50)

# Save the transpiled code to a file
with open("equinox_mlp.py", "w") as f:
    f.write(equinox_code)

print("Saved transpiled code to 'equinox_mlp.py'")

# Also test with the CNN example
pytorch_cnn_code = Path("pytorch_cnn_example.py").read_text()
equinox_cnn_code = transpiler.transpile(pytorch_cnn_code)

print("\nTranspiled CNN code using improved transpiler:")
print("-" * 50)
print(equinox_cnn_code)
print("-" * 50)

# Save the transpiled CNN code to a file
with open("equinox_cnn.py", "w") as f:
    f.write(equinox_cnn_code)

print("Saved transpiled CNN code to 'equinox_cnn.py'")
