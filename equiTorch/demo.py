from pathlib import Path
from transpiler import PyTorchToEquinoxTranspiler

pytorch_code = Path("pytorch_mlp_example.py").read_text()
transpiler = PyTorchToEquinoxTranspiler()
equinox_code = transpiler.transpile(pytorch_code)

print("Transpiled Equinox code:")
print("-" * 50)
print(equinox_code)
print("-" * 50)

with open("equinox_mlp.py", "w") as f:
    f.write(equinox_code)

print("Saved transpiled code to 'equinox_mlp.py'")
