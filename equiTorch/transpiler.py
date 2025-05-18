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
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            # For binary operations, preserve the expression
            left = self._extract_value(node.left)
            right = self._extract_value(node.right)
            
            # Determine the operator symbol
            if isinstance(node.op, ast.Mult):
                op = "*"
            elif isinstance(node.op, ast.Add):
                op = "+"
            elif isinstance(node.op, ast.Sub):
                op = "-"
            elif isinstance(node.op, ast.Div):
                op = "/"
            else:
                op = "?"
                
            # Return the expression as a string
            return f"{left} {op} {right}"
        elif isinstance(node, ast.UnaryOp):
            # For unary operations like -1
            operand = self._extract_value(node.operand)
            
            if isinstance(node.op, ast.USub):
                return -operand if isinstance(operand, (int, float)) else f"-{operand}"
            elif isinstance(node.op, ast.UAdd):
                return operand
            elif isinstance(node.op, ast.Not):
                return f"not {operand}"
            else:
                return f"?{operand}"
        elif isinstance(node, ast.List):
            # For list literals [1, 2, 3]
            elements = [self._extract_value(elt) for elt in node.elts]
            return elements
        elif isinstance(node, ast.Tuple):
            # For tuple literals (1, 2, 3)
            elements = [self._extract_value(elt) for elt in node.elts]
            return tuple(elements)
        else:
            # placeholder or more complex expressions
            return "..."
    
    def _transform_forward(self, forward_method: ast.FunctionDef) -> str:
        """
        Transform a PyTorch forward method to Equinox style by analyzing the AST.
        
        This method analyzes the abstract syntax tree (AST) of the forward method
        to generate equivalent Equinox code, preserving the order and structure
        of operations.
        
        Args:
            forward_method: AST node for the forward method
            
        Returns:
            String containing Equinox forward code
        """
        # Get the input variable name (usually 'x')
        input_var = forward_method.args.args[1].arg if len(forward_method.args.args) > 1 else 'x'
        
        # Track variable assignments to handle intermediate variables
        var_assignments = {input_var: input_var}  # Map variable names to their current value
        
        forward_body = []
        
        # Process each statement in the forward method
        for stmt in forward_method.body:
            # Skip docstrings and comments
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                continue
            
            # Handle variable assignments
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    output_var = target.id
                    
                    # Process the right side of the assignment
                    if isinstance(stmt.value, ast.Call):
                        # Special case for view operation
                        if (isinstance(stmt.value.func, ast.Attribute) and 
                            isinstance(stmt.value.func.value, ast.Name) and 
                            stmt.value.func.attr == 'view'):
                            
                            base_var = stmt.value.func.value.id
                            
                            # Handle the view operation
                            view_code = self._handle_view_operation(stmt.value, base_var, output_var)
                            if view_code:
                                forward_body.append(f"        {view_code}")
                                var_assignments[output_var] = output_var
                                continue
                        # Function call (e.g., self.fc1(x), F.relu(x), F.relu(self.fc1(x)))
                        # Check if this is a nested call like F.relu(self.fc1(x))
                        if (isinstance(stmt.value.func, ast.Attribute) and 
                            isinstance(stmt.value.func.value, ast.Name) and 
                            stmt.value.func.value.id == 'F' and 
                            len(stmt.value.args) == 1 and 
                            isinstance(stmt.value.args[0], ast.Call)):
                            
                            # This is a nested call like F.relu(self.fc1(x))
                            inner_call = stmt.value.args[0]
                            func_name = stmt.value.func.attr.lower()
                            
                            # Process the inner call first
                            if (isinstance(inner_call.func, ast.Attribute) and 
                                isinstance(inner_call.func.value, ast.Name) and 
                                inner_call.func.value.id == 'self'):
                                
                                # This is a call to a module like self.fc1(x)
                                module_name = inner_call.func.attr
                                
                                # Get the input variable(s)
                                inputs = []
                                for arg in inner_call.args:
                                    if isinstance(arg, ast.Name):
                                        arg_name = arg.id
                                        if arg_name in var_assignments:
                                            inputs.append(var_assignments[arg_name])
                                        else:
                                            inputs.append(arg_name)
                                    else:
                                        # For more complex arguments
                                        inputs.append(self._extract_value(arg))
                                
                                if inputs:
                                    input_str = ", ".join(str(i) for i in inputs)
                                    # Generate code for the nested call
                                    if func_name == 'relu':
                                        forward_body.append(f"        {output_var} = jax.nn.relu(jax.vmap(self.{module_name})({input_str}))")
                                    else:
                                        # For other functional operations
                                        forward_body.append(f"        _temp = jax.vmap(self.{module_name})({input_str})")
                                        forward_body.append(f"        {output_var} = jax.nn.{func_name}(_temp)")
                                    
                                    var_assignments[output_var] = output_var
                                    continue
                        
                        # If not a special nested call, process normally
                        code_line = self._process_call(stmt.value, output_var, var_assignments)
                        if code_line:
                            forward_body.append(f"        {code_line}")
                            var_assignments[output_var] = output_var  # Update variable mapping
                    
                    elif isinstance(stmt.value, ast.Name):
                        # Simple variable assignment (e.g., y = x)
                        input_name = stmt.value.id
                        if input_name in var_assignments:
                            forward_body.append(f"        {output_var} = {var_assignments[input_name]}")
                            var_assignments[output_var] = var_assignments[input_name]
                    
                    elif isinstance(stmt.value, ast.BinOp):
                        # Binary operation (e.g., x + y)
                        # This is a placeholder for handling binary operations
                        forward_body.append(f"        # Binary operation not fully supported yet")
                        forward_body.append(f"        {output_var} = {self._extract_value(stmt.value)}")
                
                elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    # Attribute assignment (e.g., self.x = y)
                    # Usually not common in forward methods, but handle it just in case
                    if target.value.id == 'self':
                        attr_name = target.attr
                        if isinstance(stmt.value, ast.Name):
                            input_name = stmt.value.id
                            if input_name in var_assignments:
                                forward_body.append(f"        self.{attr_name} = {var_assignments[input_name]}")
            
            # Handle return statement
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name):
                    return_var = stmt.value.id
                    if return_var in var_assignments:
                        forward_body.append(f"        return {var_assignments[return_var]}")
                    else:
                        forward_body.append(f"        return {return_var}")
                else:
                    # For more complex return expressions
                    forward_body.append(f"        return {self._extract_value(stmt.value)}")
        
        return "\n".join(forward_body)
    
    def _process_call(self, call_node: ast.Call, output_var: str, var_assignments: dict) -> str:
        """
        Process a call node and generate the corresponding Equinox code.
        
        Args:
            call_node: AST call node
            output_var: Name of the output variable
            var_assignments: Dictionary mapping variable names to their current values
            
        Returns:
            String containing Equinox code for the call
        """
        if isinstance(call_node.func, ast.Attribute):
            # Method call (e.g., self.conv1(x), F.relu(x), x.view(...))
            if isinstance(call_node.func.value, ast.Name):
                if call_node.func.value.id == 'self':
                    # Module call (e.g., self.fc1(x))
                    module_name = call_node.func.attr
                    
                    # Get the input variable(s)
                    inputs = []
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name):
                            arg_name = arg.id
                            if arg_name in var_assignments:
                                inputs.append(var_assignments[arg_name])
                            else:
                                inputs.append(arg_name)
                        else:
                            # For more complex arguments
                            inputs.append(self._extract_value(arg))
                    
                    if inputs:
                        input_str = ", ".join(str(i) for i in inputs)  # Handle multiple inputs
                        return f"{output_var} = jax.vmap(self.{module_name})({input_str})"
                    else:
                        return f"{output_var} = jax.vmap(self.{module_name})()"
                
                elif call_node.func.value.id == 'F':
                    # Functional call (e.g., F.relu(x), F.max_pool2d(x, 2))
                    func_name = call_node.func.attr.lower()
                    
                    # Get the input variable(s)
                    inputs = []
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name):
                            arg_name = arg.id
                            if arg_name in var_assignments:
                                inputs.append(var_assignments[arg_name])
                            else:
                                inputs.append(arg_name)
                        else:
                            # For more complex arguments
                            inputs.append(self._extract_value(arg))
                    
                    if not inputs:
                        return f"# Error: No inputs for {func_name}"
                    
                    # Handle different functional operations
                    if func_name == 'relu':
                        return f"{output_var} = jax.nn.relu({inputs[0]})"
                    
                    elif func_name == 'max_pool1d':
                        kernel_size = inputs[1] if len(inputs) > 1 else 2
                        stride = inputs[2] if len(inputs) > 2 else kernel_size
                        padding = inputs[3] if len(inputs) > 3 else 0
                        return f"{output_var} = jax.nn.max_pool({inputs[0]}, window_shape=({kernel_size},), strides=({stride},), padding='{padding}')"
                    
                    elif func_name == 'max_pool2d':
                        kernel_size = inputs[1] if len(inputs) > 1 else 2
                        stride = inputs[2] if len(inputs) > 2 else kernel_size
                        padding = inputs[3] if len(inputs) > 3 else 0
                        
                        # Handle different kernel_size formats
                        if isinstance(kernel_size, (list, tuple)):
                            kernel_shape = f"({kernel_size[0]}, {kernel_size[1]})"
                        else:
                            kernel_shape = f"({kernel_size}, {kernel_size})"
                            
                        # Handle different strides formats
                        if isinstance(stride, (list, tuple)):
                            strides_shape = f"({stride[0]}, {stride[1]})"
                        else:
                            strides_shape = f"({stride}, {stride})"
                        
                        return f"{output_var} = jax.nn.max_pool({inputs[0]}, window_shape={kernel_shape}, strides={strides_shape}, padding='{padding}')"
                    
                    elif func_name == 'max_pool3d':
                        kernel_size = inputs[1] if len(inputs) > 1 else 2
                        stride = inputs[2] if len(inputs) > 2 else kernel_size
                        padding = inputs[3] if len(inputs) > 3 else 0
                        
                        # Handle different kernel_size formats
                        if isinstance(kernel_size, (list, tuple)):
                            kernel_shape = f"({kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]})"
                        else:
                            kernel_shape = f"({kernel_size}, {kernel_size}, {kernel_size})"
                            
                        # Handle different strides formats
                        if isinstance(stride, (list, tuple)):
                            strides_shape = f"({stride[0]}, {stride[1]}, {stride[2]})"
                        else:
                            strides_shape = f"({stride}, {stride}, {stride})"
                        
                        return f"{output_var} = jax.nn.max_pool({inputs[0]}, window_shape={kernel_shape}, strides={strides_shape}, padding='{padding}')"
                    
                    else:
                        # Default handling for other functional operations
                        args_str = ", ".join(inputs)
                        return f"{output_var} = jax.nn.{func_name}({args_str})"
            
            elif isinstance(call_node.func.value, ast.Attribute):
                # Method call on an attribute (e.g., x.size(0), x.view(...))
                base_var = self._extract_value(call_node.func.value)
                method_name = call_node.func.attr
                
                # Handle reshape/view operations specially
                if method_name in ['view', 'reshape']:
                    # Simple approach: replace any view/reshape with jnp.reshape
                    # and convert tensor.size(dim) to tensor.shape[dim]
                    
                    # Process the shape arguments
                    shape_args = []
                    for arg in call_node.args:
                        # Special case for tensor.size(dim)
                        if (isinstance(arg, ast.Call) and 
                            isinstance(arg.func, ast.Attribute) and 
                            arg.func.attr == 'size' and 
                            len(arg.args) == 1 and 
                            isinstance(arg.args[0], ast.Constant)):
                            
                            dim_index = arg.args[0].value
                            shape_args.append(f"{base_var}.shape[{dim_index}]")
                        else:
                            # For other arguments
                            shape_args.append(str(self._extract_value(arg)))
                    
                    # Join the shape arguments
                    shape_str = ", ".join(shape_args)
                    return f"{output_var} = jnp.reshape({base_var}, ({shape_str}))"
                else:
                    # For other method calls (e.g., x.size(0))
                    # Get the arguments
                    args = []
                    for arg in call_node.args:
                        args.append(self._extract_value(arg))
                    
                    args_str = ", ".join(args)
                    return f"{output_var} = {base_var}.{method_name}({args_str})"
            
            elif isinstance(call_node.func.value, ast.Call):
                # Nested method call (e.g., self.fc1(x).relu())
                # This is a complex case, handle it recursively
                temp_var = f"_temp_{output_var}"
                inner_call = self._process_call(call_node.func.value, temp_var, var_assignments)
                method_name = call_node.func.attr
                
                # Get the arguments
                args = []
                for arg in call_node.args:
                    args.append(self._extract_value(arg))
                
                args_str = ", ".join(args)
                
                if inner_call:
                    return f"{inner_call}\n        {output_var} = {temp_var}.{method_name}({args_str})"
                else:
                    return f"# Error: Could not process nested call"
        
        elif isinstance(call_node.func, ast.Name):
            # Direct function call (e.g., relu(x))
            func_name = call_node.func.id
            
            # Get the input variable(s)
            inputs = []
            for arg in call_node.args:
                if isinstance(arg, ast.Name):
                    arg_name = arg.id
                    if arg_name in var_assignments:
                        inputs.append(var_assignments[arg_name])
                    else:
                        inputs.append(arg_name)
                else:
                    # For more complex arguments
                    inputs.append(self._extract_value(arg))
            
            args_str = ", ".join(inputs)
            return f"{output_var} = {func_name}({args_str})"
        
        elif isinstance(call_node.func, ast.Call):
            # Nested function call (e.g., F.relu(self.fc1(x)))
            # This is a complex case, handle it recursively
            temp_var = f"_temp_{output_var}"
            inner_call = self._process_call(call_node.func, temp_var, var_assignments)
            
            # Get the arguments
            args = []
            for arg in call_node.args:
                args.append(self._extract_value(arg))
            
            args_str = ", ".join(args)
            
            if inner_call:
                return f"{inner_call}\n        {output_var} = {temp_var}({args_str})"
            else:
                return f"# Error: Could not process nested call"
        
        return f"# Unsupported call type: {ast.dump(call_node)}"
    
    def _handle_view_operation(self, view_call: ast.Call, base_var: str, output_var: str) -> str:
        """
        Handle the view operation by converting it to jnp.reshape.
        
        Args:
            view_call: AST call node for the view operation
            base_var: Name of the variable the view is applied to
            output_var: Name of the output variable
            
        Returns:
            String containing Equinox code for the view operation
        """
        # Process the shape arguments
        shape_args = []
        for arg in view_call.args:
            # Special case for tensor.size(dim)
            if (isinstance(arg, ast.Call) and 
                isinstance(arg.func, ast.Attribute) and 
                arg.func.attr == 'size' and 
                len(arg.args) == 1 and 
                isinstance(arg.args[0], ast.Constant)):
                
                dim_index = arg.args[0].value
                shape_args.append(f"{base_var}.shape[{dim_index}]")
            elif (isinstance(arg, ast.UnaryOp) and 
                  isinstance(arg.op, ast.USub) and 
                  isinstance(arg.operand, ast.Constant) and 
                  arg.operand.value == 1):
                # Handle -1 in reshape
                shape_args.append("-1")
            else:
                # For other arguments
                shape_args.append(str(self._extract_value(arg)))
        
        # Join the shape arguments
        shape_str = ", ".join(shape_args)
        return f"{output_var} = jnp.reshape({base_var}, ({shape_str}))"
    
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
