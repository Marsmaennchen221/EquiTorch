from pathlib import Path
from graph_transpiler import PyTorchToEquinoxGraphTranspiler

# Read the PyTorch MLP example code
pytorch_code = Path("pytorch_mlp_example.py").read_text()

# Create a graph-based transpiler
transpiler = PyTorchToEquinoxGraphTranspiler()

# Transpile the PyTorch code to Equinox code
equinox_code = transpiler.transpile(pytorch_code)

print("Transpiled Equinox code using graph-based transpiler:")
print("-" * 50)
print(equinox_code)
print("-" * 50)

# Save the transpiled code to a file
with open("equinox_mlp_graph.py", "w") as f:
    f.write(equinox_code)

print("Saved transpiled code to 'equinox_mlp_graph.py'")

# Also test with the CNN example
pytorch_cnn_code = Path("pytorch_cnn_example.py").read_text()
equinox_cnn_code = transpiler.transpile(pytorch_cnn_code)

print("\nTranspiled CNN code using graph-based transpiler:")
print("-" * 50)
print(equinox_cnn_code)
print("-" * 50)

# Save the transpiled CNN code to a file
with open("equinox_cnn_graph.py", "w") as f:
    f.write(equinox_cnn_code)

print("Saved transpiled CNN code to 'equinox_cnn_graph.py'")
