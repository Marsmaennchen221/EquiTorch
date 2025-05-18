import ast
import pprint

# Read the PyTorch CNN example code
with open("pytorch_cnn_example.py", "r") as f:
    pytorch_code = f.read()

# Parse the code into an AST
tree = ast.parse(pytorch_code)

# Find the view operation
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'view':
        print("Found view operation:")
        print(ast.dump(node, indent=4))
        
        # Print the arguments
        print("\nArguments:")
        for i, arg in enumerate(node.args):
            print(f"Argument {i}:")
            print(ast.dump(arg, indent=4))
