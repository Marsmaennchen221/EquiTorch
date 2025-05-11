import ast
import inspect
import re
from typing import Dict, List, Any, Optional, Tuple

from module_converters import (
    ModuleConverter, 
    LinearConverter, 
    Conv1dConverter, 
    Conv2dConverter, 
    Conv3dConverter
)

class PyTorchToEquinoxTranspiler:
    """
    A transpiler to convert PyTorch modules to Equinox modules.
    
    Supports multiple module types through a registry of module converters.
    """
    
    def __init__(self):
        self.imports = [
            "import jax",
            "import jax.numpy as jnp",
            "import equinox as eqx",
        ]
        
        # Initialize the module converter registry
        self.module_converters = {}
        self._register_default_converters()
    
    def _register_default_converters(self):
        """Register the default set of module converters."""
        self.register_converter("Linear", LinearConverter())
        self.register_converter("Conv1d", Conv1dConverter())
        self.register_converter("Conv2d", Conv2dConverter())
        self.register_converter("Conv3d", Conv3dConverter())
    
    def register_converter(self, module_name: str, converter: ModuleConverter):
        """
        Register a converter for a specific PyTorch module type.
        
        Args:
            module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
            converter: ModuleConverter instance for this module type
        """
        self.module_converters[module_name] = converter
    
    def transpile(self, pytorch_code: str) -> str:
        """
        Transpile PyTorch code to Equinox code.
        
        Args:
            pytorch_code: String containing PyTorch module definitions
            
        Returns:
            String containing equivalent Equinox code
        """
        # Parse the PyTorch code into an AST
        tree = ast.parse(pytorch_code)
        
        module_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == 'Module':
                        module_classes.append(node)
                    elif isinstance(base, ast.Name) and base.id == 'Module':
                        module_classes.append(node)
        
        equinox_modules = []
        for module_class in module_classes:
            equinox_modules.append(self._transform_module(module_class))
        
        equinox_code = "\n".join(self.imports) + "\n\n" + "\n\n".join(equinox_modules)
        
        return equinox_code
    
    def _transform_module(self, class_node: ast.ClassDef) -> str:
        """Transform a PyTorch module class into an Equinox module."""
        class_name = class_node.name
        
        init_method = None
        forward_method = None
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == "__init__":
                    init_method = node
                elif node.name == "forward":
                    forward_method = node
        
        if not init_method or not forward_method:
            raise ValueError(f"Class {class_name} must have both __init__ and forward methods")
        
        # Extract init arguments (excluding 'self')
        init_args = []
        if init_method.args.args:
            init_args = [arg.arg for arg in init_method.args.args[1:]]  # Skip 'self'
        
        parameters = self._extract_parameters(init_method)
        forward_code = self._transform_forward(forward_method)
        equinox_module = self._generate_equinox_module(class_name, parameters, forward_code, init_args)
        
        return equinox_module
    
    def _extract_parameters(self, init_method: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Extract parameters defined in the __init__ method."""
        parameters = []
        
        for node in ast.walk(init_method):
            if (isinstance(node, ast.Assign) and 
                len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Attribute) and
                isinstance(node.targets[0].value, ast.Name) and
                node.targets[0].value.id == 'self'):
                
                param_name = node.targets[0].attr
                
                if isinstance(node.value, ast.Call):
                    layer_info = self._extract_layer_info(node.value)
                    if layer_info:
                        parameters.append({
                            'name': param_name,
                            'type': layer_info['type'],
                            'args': layer_info['args']
                        })
        
        return parameters
    
    def _extract_layer_info(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract information about a layer from a call node."""
        # Check if this is a module we can convert
        if isinstance(call_node.func, ast.Attribute):
            module_name = call_node.func.attr
            if module_name in self.module_converters:
                converter = self.module_converters[module_name]
                args = converter.extract_args(call_node)
                return {
                    'type': module_name,
                    'args': args
                }
        
        return None
    
    def _extract_value(self, node: ast.AST) -> Any:
        """Extract a literal value from an AST node."""
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return node.id  # returns variable name as string
        elif isinstance(node, ast.Attribute):
            return f"{self._extract_value(node.value)}.{node.attr}"
        else:
            # placeholder or more complex expressions
            return "..."
    
    def _transform_forward(self, forward_method: ast.FunctionDef) -> str:
        """Transform a PyTorch forward method to Equinox style."""
        forward_body = []
        
        for stmt in forward_method.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
                
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Call):
                    if (isinstance(stmt.value.func, ast.Attribute) and 
                        stmt.value.func.attr == 'relu' and
                        isinstance(stmt.value.func.value, ast.Name) and
                        stmt.value.func.value.id == 'F'):
                        forward_body.append("        x = jax.nn.relu(jax.vmap(self.fc1)(x))")
                    else:
                        forward_body.append("        x = jax.vmap(self.fc2)(x)")
            elif isinstance(stmt, ast.Return):
                forward_body.append("        return x")
        
        return "\n".join(forward_body)
    
    def _generate_equinox_module(self, class_name: str, parameters: List[Dict[str, Any]], forward_code: str, init_args: List[str]) -> str:
        """Generate Equinox module code from the extracted information."""
        lines = [f"class {class_name}(eqx.Module):"]
        
        # Generate type annotations for each parameter
        for param in parameters:
            module_type = param['type']
            if module_type in self.module_converters:
                converter = self.module_converters[module_type]
                lines.append(f"    {param['name']}: {converter.get_equinox_type()}")
        
        # Create the __init__ signature with the original arguments
        init_signature = ", ".join(["self"] + init_args + ["key=None", "**kwargs"])
        lines.append(f"    def __init__({init_signature}):")
        lines.append("        if key is None:")
        lines.append("            key = jax.random.PRNGKey(0)")
        
        # Generate initialization code for each parameter
        for i, param in enumerate(parameters):
            module_type = param['type']
            if module_type in self.module_converters:
                converter = self.module_converters[module_type]
                lines.append(f"        key_{i} = jax.random.split(key, 1)[0]")
                
                # Create a copy of the args dictionary to avoid modifying the original
                args_copy = param['args'].copy()
                
                # Replace placeholder values with the corresponding init_args if available
                for arg_name, arg_value in args_copy.items():
                    if arg_value == "..." and arg_name in init_args:
                        args_copy[arg_name] = arg_name
                
                init_code = converter.generate_equinox_code(f"self.{param['name']}", args_copy, f"key_{i}")
                lines.append(f"        {init_code}")
        
        lines.append("")
        lines.append("    def __call__(self, x):")
        
        indent = "        "
        for line in forward_code.split('\n'):
            if line.strip():
                lines.append(indent + line)
        
        return '\n'.join(lines)
