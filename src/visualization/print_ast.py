"""Visualization for the AST."""
import ast
from pathlib import Path

# Read the PyTorch CNN example code
with Path("../../tests/networks/pytorch_cnn_example.py").open("r") as f:
    pytorch_code = f.read()

# Parse the code into an AST
tree = ast.parse(pytorch_code)

# Find the view operation
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'view':
        print("Found view operation:")  # noqa: T201
        print(ast.dump(node, indent=4))  # noqa: T201
        
        # Print the arguments
        print("\nArguments:")  # noqa: T201
        for i, arg in enumerate(node.args):
            print(f"Argument {i}:")  # noqa: T201
            print(ast.dump(arg, indent=4))  # noqa: T201
