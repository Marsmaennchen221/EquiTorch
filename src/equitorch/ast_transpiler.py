"""
AST-based PyTorch to Equinox transpiler.

This module provides an AST-based transpiler that constructs syntax trees
instead of generating code through string concatenation.
"""

import ast
import sys
from typing import Dict, List, Any, Optional

from .module_converters import (
    ModuleConverter, 
    LinearConverter, 
    Conv1dConverter, 
    Conv2dConverter, 
    Conv3dConverter
)
from .ast_builders import (
    EquinoxASTBuilder,
    EquinoxClassBuilder,
    EquinoxCallBuilder,
    EquinoxModuleBuilder
)


class ASTBasedTranspiler:
    """
    An AST-based transpiler to convert PyTorch modules to Equinox modules.
    
    This transpiler constructs AST nodes instead of generating strings,
    providing better correctness guarantees and maintainability.
    """
    
    def __init__(self):
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
        Transpile PyTorch code to Equinox code using AST construction.
        
        Args:
            pytorch_code: String containing PyTorch module definitions
            
        Returns:
            String containing equivalent Equinox code
        """
        # Parse the PyTorch code into an AST
        tree = ast.parse(pytorch_code)
        
        # Find all PyTorch module classes
        module_classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == 'Module':
                        module_classes.append(node)
                    elif isinstance(base, ast.Name) and base.id == 'Module':
                        module_classes.append(node)
        
        # Build the Equinox module AST
        module_builder = EquinoxModuleBuilder()
        
        # Add standard imports
        module_builder.add_import("jax")
        module_builder.add_import("jax.numpy", "jnp")
        module_builder.add_import("equinox", "eqx")
        
        # Transform each PyTorch module class
        for module_class in module_classes:
            equinox_class = self._transform_module_to_ast(module_class)
            module_builder.add_class(equinox_class)
        
        # Build the complete module AST
        equinox_ast = module_builder.build()
        
        # Fix missing locations (lineno, col_offset) for all nodes
        ast.fix_missing_locations(equinox_ast)
        
        # Unparse the AST to generate code
        if sys.version_info >= (3, 9):
            return ast.unparse(equinox_ast)
        else:
            # Fallback for older Python versions
            try:
                import astor
                return astor.to_source(equinox_ast)
            except ImportError:
                raise RuntimeError("Python 3.9+ required for ast.unparse, or install 'astor' package for older versions")
    
    def _transform_module_to_ast(self, class_node: ast.ClassDef) -> ast.ClassDef:
        """Transform a PyTorch module class into an Equinox module AST."""
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
        
        # Extract parameters from __init__ method
        parameters = self._extract_parameters(init_method)
        
        # Build the Equinox class
        class_builder = EquinoxClassBuilder(class_name)
        
        # Add init arguments
        for arg in init_args:
            class_builder.add_init_arg(arg)
        
        # Add type annotations
        for param in parameters:
            module_type = param['type']
            if module_type in self.module_converters:
                converter = self.module_converters[module_type]
                class_builder.add_type_annotation(param['name'], converter.get_equinox_type())
        
        # Add key initialization check
        key_check = ast.If(
            test=ast.Compare(
                left=EquinoxASTBuilder.create_name('key'),
                ops=[ast.Is()],
                comparators=[EquinoxASTBuilder.create_constant(None)]
            ),
            body=[
                EquinoxASTBuilder.create_assign(
                    'key',
                    EquinoxASTBuilder.create_call(
                        'jax.random.PRNGKey',
                        [EquinoxASTBuilder.create_constant(0)]
                    )
                )
            ],
            orelse=[]
        )
        class_builder.add_init_statement(key_check)
        
        # Add parameter initialization
        for i, param in enumerate(parameters):
            module_type = param['type']
            if module_type in self.module_converters:
                converter = self.module_converters[module_type]
                
                # Generate key split
                key_split = EquinoxASTBuilder.create_assign(
                    f'key_{i}',
                    ast.Subscript(
                        value=EquinoxASTBuilder.create_call(
                            'jax.random.split',
                            [EquinoxASTBuilder.create_name('key'), EquinoxASTBuilder.create_constant(1)]
                        ),
                        slice=EquinoxASTBuilder.create_constant(0),
                        ctx=ast.Load()
                    )
                )
                class_builder.add_init_statement(key_split)
                
                # Create a copy of the args dictionary to avoid modifying the original
                args_copy = param['args'].copy()
                
                # Replace placeholder values with the corresponding init_args if available
                for arg_name, arg_value in args_copy.items():
                    if arg_value == "..." and arg_name in init_args:
                        args_copy[arg_name] = arg_name
                
                # Generate parameter initialization
                param_init = converter.generate_equinox_ast(f"self.{param['name']}", args_copy, f"key_{i}")
                class_builder.add_init_statement(param_init)
        
        # Transform forward method
        forward_statements = self._transform_forward_to_ast(forward_method)
        for stmt in forward_statements:
            class_builder.add_call_statement(stmt)
        
        return class_builder.build()
    
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
    
    def _transform_forward_to_ast(self, forward_method: ast.FunctionDef) -> List[ast.AST]:
        """
        Transform a PyTorch forward method to Equinox AST nodes.
        
        Args:
            forward_method: AST node for the forward method
            
        Returns:
            List of AST statements for the Equinox forward method
        """
        # Get the input variable name (usually 'x')
        input_var = forward_method.args.args[1].arg if len(forward_method.args.args) > 1 else 'x'
        
        # Track variable assignments to handle intermediate variables
        var_assignments = {input_var: input_var}  # Map variable names to their current value
        
        forward_statements = []
        
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
                        # First check if this is a nested call like F.relu(self.conv1(x))
                        nested_stmt = self._handle_nested_call(stmt.value, output_var, var_assignments)
                        if nested_stmt:
                            if isinstance(nested_stmt, list):
                                forward_statements.extend(nested_stmt)
                            else:
                                forward_statements.append(nested_stmt)
                            var_assignments[output_var] = output_var
                        else:
                            # Handle regular calls
                            ast_stmt = self._process_call_to_ast(stmt.value, output_var, var_assignments)
                            if ast_stmt:
                                forward_statements.append(ast_stmt)
                                var_assignments[output_var] = output_var
                    
                    elif isinstance(stmt.value, ast.Name):
                        # Simple variable assignment (e.g., y = x)
                        input_name = stmt.value.id
                        if input_name in var_assignments:
                            assign_stmt = EquinoxASTBuilder.create_assign(
                                output_var, 
                                EquinoxASTBuilder.create_name(var_assignments[input_name])
                            )
                            forward_statements.append(assign_stmt)
                            var_assignments[output_var] = var_assignments[input_name]
            
            # Handle return statement
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Name):
                    return_var = stmt.value.id
                    if return_var in var_assignments:
                        return_stmt = ast.Return(value=EquinoxASTBuilder.create_name(var_assignments[return_var]))
                    else:
                        return_stmt = ast.Return(value=EquinoxASTBuilder.create_name(return_var))
                    forward_statements.append(return_stmt)
        
        return forward_statements
    
    def _process_call_to_ast(self, call_node: ast.Call, output_var: str, var_assignments: dict) -> Optional[ast.AST]:
        """
        Process a call node and generate the corresponding Equinox AST.
        
        Args:
            call_node: AST call node
            output_var: Name of the output variable
            var_assignments: Dictionary mapping variable names to their current values
            
        Returns:
            AST statement for the Equinox call
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
                    
                    if inputs:
                        # Create jax.vmap(self.module)(input) call
                        vmap_call = EquinoxCallBuilder.create_jax_vmap_call(module_name, inputs[0])
                        return EquinoxASTBuilder.create_assign(output_var, vmap_call)
                
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
                            # For more complex arguments, extract the value
                            inputs.append(self._extract_value_from_ast(arg))
                    
                    if not inputs:
                        return None
                    
                    # Handle different functional operations
                    if func_name == 'relu':
                        relu_call = EquinoxCallBuilder.create_jax_nn_call('relu', inputs[0])
                        return EquinoxASTBuilder.create_assign(output_var, relu_call)
                    
                    elif func_name in ['max_pool1d', 'max_pool2d', 'max_pool3d']:
                        # Extract pooling parameters
                        kernel_size = inputs[1] if len(inputs) > 1 else 2
                        stride = inputs[2] if len(inputs) > 2 else kernel_size
                        padding = inputs[3] if len(inputs) > 3 else 0
                        
                        # Convert to appropriate format
                        if func_name == 'max_pool1d':
                            if isinstance(kernel_size, (int, float)):
                                window_shape = (kernel_size,)
                                strides = (stride,)
                            else:
                                window_shape = kernel_size
                                strides = stride
                        elif func_name == 'max_pool2d':
                            if isinstance(kernel_size, (int, float)):
                                window_shape = (kernel_size, kernel_size)
                                strides = (stride, stride)
                            else:
                                window_shape = kernel_size
                                strides = stride
                        elif func_name == 'max_pool3d':
                            if isinstance(kernel_size, (int, float)):
                                window_shape = (kernel_size, kernel_size, kernel_size)
                                strides = (stride, stride, stride)
                            else:
                                window_shape = kernel_size
                                strides = stride
                        
                        pool_call = EquinoxCallBuilder.create_jax_nn_call(
                            'max_pool', 
                            inputs[0],
                            window_shape=window_shape,
                            strides=strides,
                            padding=str(padding)
                        )
                        return EquinoxASTBuilder.create_assign(output_var, pool_call)
                
                else:
                    # Method call on a variable (e.g., x.view(...))
                    base_var = call_node.func.value.id
                    method_name = call_node.func.attr
                    
                    # Handle reshape/view operations
                    if method_name in ['view', 'reshape']:
                        # Process the shape arguments
                        shape_args = []
                        for arg in call_node.args:
                            shape_args.append(self._extract_value_from_ast(arg))
                        
                        reshape_call = EquinoxCallBuilder.create_jnp_reshape_call(base_var, shape_args)
                        return EquinoxASTBuilder.create_assign(output_var, reshape_call)
            
            elif isinstance(call_node.func.value, ast.Attribute):
                # Method call on an attribute (e.g., self.something.view(...))
                base_var = self._extract_value_from_ast(call_node.func.value)
                method_name = call_node.func.attr
                
                # Handle reshape/view operations
                if method_name in ['view', 'reshape']:
                    # Process the shape arguments
                    shape_args = []
                    for arg in call_node.args:
                        shape_args.append(self._extract_value_from_ast(arg))
                    
                    reshape_call = EquinoxCallBuilder.create_jnp_reshape_call(base_var, shape_args)
                    return EquinoxASTBuilder.create_assign(output_var, reshape_call)
            
        
        return None
    
    def _extract_value_from_ast(self, node: ast.AST) -> Any:
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
            base = self._extract_value_from_ast(node.value)
            return f"{base}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Call):
            # Handle special cases like tensor.size(dim)
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr == 'size' and 
                len(node.args) == 1 and 
                isinstance(node.args[0], ast.Constant)):
                
                base_var = self._extract_value_from_ast(node.func.value)
                dim_index = node.args[0].value
                return f"{base_var}.shape[{dim_index}]"
        elif isinstance(node, ast.UnaryOp):
            # For unary operations like -1
            operand = self._extract_value_from_ast(node.operand)
            
            if isinstance(node.op, ast.USub):
                return -operand if isinstance(operand, (int, float)) else f"-{operand}"
            elif isinstance(node.op, ast.UAdd):
                return operand
            elif isinstance(node.op, ast.Not):
                return f"not {operand}"
        
        # Default fallback
        return "..."
    
    def _handle_nested_call(self, call_node: ast.Call, output_var: str, var_assignments: dict) -> Optional[ast.AST]:
        """
        Handle nested calls like F.relu(self.conv1(x)).
        
        Args:
            call_node: AST call node
            output_var: Name of the output variable
            var_assignments: Dictionary mapping variable names to their current values
            
        Returns:
            AST statement for the nested call
        """
        # Check if this is a nested call like F.relu(self.conv1(x))
        if (isinstance(call_node.func, ast.Attribute) and 
            isinstance(call_node.func.value, ast.Name) and 
            call_node.func.value.id == 'F' and 
            len(call_node.args) == 1 and 
            isinstance(call_node.args[0], ast.Call)):
            
            # This is a nested call like F.relu(self.conv1(x))
            inner_call = call_node.args[0]
            func_name = call_node.func.attr.lower()
            
            # Process the inner call first
            if (isinstance(inner_call.func, ast.Attribute) and 
                isinstance(inner_call.func.value, ast.Name) and 
                inner_call.func.value.id == 'self'):
                
                # This is a call to a module like self.conv1(x)
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
                
                if inputs:
                    # Generate code for the nested call
                    if func_name == 'relu':
                        # Create jax.nn.relu(jax.vmap(self.module)(input))
                        vmap_call = EquinoxCallBuilder.create_jax_vmap_call(module_name, inputs[0])
                        relu_call = EquinoxCallBuilder.create_jax_nn_call('relu', 'temp_var')
                        
                        # We need to create the vmap call result first, then apply relu
                        # For simplicity, we'll create a combined call
                        combined_call = EquinoxASTBuilder.create_call(
                            'jax.nn.relu',
                            [vmap_call]
                        )
                        return EquinoxASTBuilder.create_assign(output_var, combined_call)
        
        return None
