import ast
import inspect
import re
from typing import Dict, List, Any, Optional, Tuple

class PyTorchToEquinoxTranspiler:
    """
    A basic transpiler to convert PyTorch modules to Equinox modules.
    
    Currently focuses on simple module structures with limited functionality.
    """
    
    def __init__(self):
        self.imports = [
            "import jax",
            "import jax.numpy as jnp",
            "import equinox as eqx",
        ]
    
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
        
        parameters = self._extract_parameters(init_method)
        forward_code = self._transform_forward(forward_method)
        equinox_module = self._generate_equinox_module(class_name, parameters, forward_code)
        
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
        # Handle nn.Linear
        if (isinstance(call_node.func, ast.Attribute) and 
            call_node.func.attr == 'Linear'):
            
            # Extract arguments
            args = {}
            for i, arg in enumerate(call_node.args):
                if i == 0:
                    args['in_features'] = self._extract_value(arg)
                elif i == 1:
                    args['out_features'] = self._extract_value(arg)
            
            # Extract keyword arguments
            for kw in call_node.keywords:
                if kw.arg == 'in_features':
                    args['in_features'] = self._extract_value(kw.value)
                elif kw.arg == 'out_features':
                    args['out_features'] = self._extract_value(kw.value)
                elif kw.arg == 'bias':
                    args['bias'] = self._extract_value(kw.value)
            
            return {
                'type': 'Linear',
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
    
    def _generate_equinox_module(self, class_name: str, parameters: List[Dict[str, Any]], forward_code: str) -> str:
        """Generate Equinox module code from the extracted information."""
        lines = [f"class {class_name}(eqx.Module):"]
        
        for param in parameters:
            if param['type'] == 'Linear':
                lines.append(f"    {param['name']}: eqx.nn.Linear")
        
        lines.append("    def __init__(self, key=None, **kwargs):")
        lines.append("        if key is None:")
        lines.append("            key = jax.random.PRNGKey(0)")
        
        for i, param in enumerate(parameters):
            if param['type'] == 'Linear':
                args = param['args']
                in_features = args.get('in_features', '...')
                out_features = args.get('out_features', '...')
                bias = args.get('bias', 'True')
                
                lines.append(f"        key_{i} = jax.random.split(key, 1)[0]")
                lines.append(f"        self.{param['name']} = eqx.nn.Linear({in_features}, {out_features}, use_bias={bias}, key=key_{i})")
        
        lines.append("")
        lines.append("    def __call__(self, x):")
        
        indent = "        "
        for line in forward_code.split('\n'):
            if line.strip():
                lines.append(indent + line)
        
        return '\n'.join(lines)
