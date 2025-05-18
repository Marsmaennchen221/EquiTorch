import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import re

from module_converters import (
    ModuleConverter, 
    LinearConverter, 
    Conv1dConverter, 
    Conv2dConverter, 
    Conv3dConverter
)

class OperationNode:
    """
    Represents a node in the computation graph.
    
    Each node corresponds to an operation in the forward pass.
    """
    
    def __init__(self, 
                 operation_type: str, 
                 inputs: List[Union[str, 'OperationNode']], 
                 output: str,
                 attributes: Dict[str, Any] = None):
        """
        Initialize an operation node.
        
        Args:
            operation_type: Type of operation (e.g., 'linear', 'relu', 'conv2d')
            inputs: List of input variable names or nodes
            output: Output variable name
            attributes: Additional attributes for the operation (e.g., kernel_size for conv)
        """
        self.operation_type = operation_type
        self.inputs = inputs
        self.output = output
        self.attributes = attributes or {}
    
    def __repr__(self):
        return f"OperationNode({self.operation_type}, inputs={self.inputs}, output={self.output})"


class ComputationGraph:
    """
    Represents a computation graph for a forward pass.
    
    The graph consists of operation nodes connected by variable dependencies.
    """
    
    def __init__(self):
        self.nodes = []
        self.input_vars = set()
        self.output_var = None
    
    def add_node(self, node: OperationNode):
        """Add a node to the graph."""
        self.nodes.append(node)
    
    def set_input_vars(self, input_vars: Set[str]):
        """Set the input variables for the graph."""
        self.input_vars = input_vars
    
    def set_output_var(self, output_var: str):
        """Set the output variable for the graph."""
        self.output_var = output_var


class PyTorchToEquinoxGraphTranspiler:
    """
    A transpiler that uses computation graphs to convert PyTorch modules to Equinox modules.
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
        
        # Initialize the operation converter registry
        self.operation_converters = {}
        self._register_default_operation_converters()
    
    def _register_default_converters(self):
        """Register the default set of module converters."""
        self.register_converter("Linear", LinearConverter())
        self.register_converter("Conv1d", Conv1dConverter())
        self.register_converter("Conv2d", Conv2dConverter())
        self.register_converter("Conv3d", Conv3dConverter())
    
    def _register_default_operation_converters(self):
        """Register the default set of operation converters."""
        # PyTorch module operations
        self.operation_converters["linear"] = self._convert_linear_op
        self.operation_converters["conv1d"] = self._convert_conv1d_op
        self.operation_converters["conv2d"] = self._convert_conv2d_op
        self.operation_converters["conv3d"] = self._convert_conv3d_op
        
        # PyTorch functional operations
        self.operation_converters["relu"] = self._convert_relu_op
        self.operation_converters["max_pool1d"] = self._convert_max_pool1d_op
        self.operation_converters["max_pool2d"] = self._convert_max_pool2d_op
        self.operation_converters["max_pool3d"] = self._convert_max_pool3d_op
        self.operation_converters["view"] = self._convert_view_op
        self.operation_converters["reshape"] = self._convert_reshape_op
    
    def register_converter(self, module_name: str, converter: ModuleConverter):
        """
        Register a converter for a specific PyTorch module type.
        
        Args:
            module_name: Name of the PyTorch module (e.g., 'Linear', 'Conv2d')
            converter: ModuleConverter instance for this module type
        """
        self.module_converters[module_name] = converter
    
    def register_operation_converter(self, operation_type: str, converter_func):
        """
        Register a converter function for a specific operation type.
        
        Args:
            operation_type: Type of operation (e.g., 'relu', 'max_pool2d')
            converter_func: Function that converts the operation to Equinox code
        """
        self.operation_converters[operation_type] = converter_func
    
    def validate_graph(self, graph: ComputationGraph) -> Tuple[bool, Optional[str]]:
        """
        Validate that all operations in the graph can be transpiled.
        
        Args:
            graph: Computation graph to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for node in graph.nodes:
            if node.operation_type not in self.operation_converters:
                return False, f"Unsupported operation: {node.operation_type}"
        
        return True, None
    
    def transpile(self, pytorch_code: str) -> str:
        """
        Transpile PyTorch code to Equinox code.
        
        Args:
            pytorch_code: String containing PyTorch module definitions
            
        Returns:
            String containing equivalent Equinox code
            
        Raises:
            ValueError: If the transpilation process encounters an unsupported operation
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
        """
        Transform a PyTorch module class into an Equinox module.
        
        Args:
            class_node: AST node for the class definition
            
        Returns:
            String containing the Equinox module code
            
        Raises:
            ValueError: If the module contains unsupported operations
        """
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
        
        # Build computation graph from forward method
        graph = self._build_computation_graph(forward_method)
        
        # Validate the graph before proceeding
        is_valid, error_message = self.validate_graph(graph)
        if not is_valid:
            raise ValueError(f"Transpilation aborted for {class_name}: {error_message}")
        
        # Generate forward code from computation graph
        forward_code = self._generate_forward_code_from_graph(graph)
        
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
        else:
            # placeholder for more complex expressions
            return "..."
    
    def _build_computation_graph(self, forward_method: ast.FunctionDef) -> ComputationGraph:
        """
        Build a computation graph from a PyTorch forward method.
        
        Args:
            forward_method: AST node for the forward method
            
        Returns:
            ComputationGraph representing the forward pass
        """
        graph = ComputationGraph()
        
        # Extract the input variable name (usually 'x')
        input_var = forward_method.args.args[1].arg if len(forward_method.args.args) > 1 else 'x'
        graph.set_input_vars({input_var})
        
        # Track variable assignments
        var_assignments = {}
        
        # Process each statement in the forward method
        for stmt in forward_method.body:
            if isinstance(stmt, ast.Assign):
                # Handle variable assignment
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        output_var = target.id
                        
                        # Handle different types of assignments
                        if isinstance(stmt.value, ast.Call):
                            self._process_call_node(stmt.value, output_var, graph, var_assignments)
                        elif isinstance(stmt.value, ast.BinOp):
                            # Handle binary operations (e.g., x + y)
                            pass  # To be implemented
                        elif isinstance(stmt.value, ast.Name):
                            # Simple variable assignment (e.g., y = x)
                            var_assignments[output_var] = stmt.value.id
            
            elif isinstance(stmt, ast.Expr):
                # Handle expression statements (might be function calls without assignment)
                if isinstance(stmt.value, ast.Call):
                    # For now, we'll just log these as comments in the graph
                    # In a real implementation, you might want to handle these differently
                    pass
            
            elif isinstance(stmt, ast.Return):
                # Handle return statement
                if isinstance(stmt.value, ast.Name):
                    graph.set_output_var(stmt.value.id)
                # Handle other return value types if needed
        
        # Process comments in the forward method to extract additional information
        for i, stmt in enumerate(forward_method.body):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                # This is a comment, check if it contains shape information
                comment = stmt.value.value
                if "->" in comment and "[" in comment and "]" in comment:
                    # This looks like a shape comment, extract information if needed
                    pass
        
        return graph
    
    def _process_call_node(self, call_node: ast.Call, output_var: str, 
                           graph: ComputationGraph, var_assignments: Dict[str, str]):
        """
        Process a call node and add the corresponding operation to the graph.
        
        Args:
            call_node: AST call node
            output_var: Name of the output variable
            graph: Computation graph to update
            var_assignments: Dictionary of variable assignments
        """
        # Handle different types of calls
        if isinstance(call_node.func, ast.Attribute):
            # Method call (e.g., self.conv1(x), F.relu(x))
            if isinstance(call_node.func.value, ast.Name):
                if call_node.func.value.id == 'self':
                    # Module call (e.g., self.conv1(x))
                    module_name = call_node.func.attr
                    inputs = []
                    
                    # Get the input variable(s)
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name):
                            inputs.append(arg.id)
                    
                    # Determine the operation type based on the module name
                    # This is a simplification; in a real implementation, you'd need to track
                    # the types of modules defined in __init__
                    operation_type = module_name.lower()
                    if operation_type.startswith('conv'):
                        if '1d' in module_name:
                            operation_type = 'conv1d'
                        elif '2d' in module_name:
                            operation_type = 'conv2d'
                        elif '3d' in module_name:
                            operation_type = 'conv3d'
                    elif operation_type.startswith('fc') or operation_type.startswith('linear'):
                        operation_type = 'linear'
                    
                    # Create and add the operation node
                    # Include the module attribute name for better code generation
                    attributes = {'module_attr': module_name}
                    node = OperationNode(operation_type, inputs, output_var, attributes)
                    graph.add_node(node)
                
                elif call_node.func.value.id == 'F':
                    # Functional call (e.g., F.relu(x), F.max_pool2d(x, 2))
                    func_name = call_node.func.attr.lower()
                    inputs = []
                    
                    # Get the input variable(s)
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name):
                            inputs.append(arg.id)
                    
                    # Extract additional attributes from keywords and positional args
                    attributes = {}
                    
                    # Handle positional arguments for common functions
                    if func_name == 'max_pool2d' and len(call_node.args) > 1:
                        # Second arg is kernel_size
                        if len(call_node.args) > 1:
                            attributes['kernel_size'] = self._extract_value(call_node.args[1])
                        # Third arg is stride (if provided)
                        if len(call_node.args) > 2:
                            attributes['stride'] = self._extract_value(call_node.args[2])
                        # Fourth arg is padding (if provided)
                        if len(call_node.args) > 3:
                            attributes['padding'] = self._extract_value(call_node.args[3])
                    
                    # Extract keyword arguments
                    for kw in call_node.keywords:
                        attributes[kw.arg] = self._extract_value(kw.value)
                    
                    # Create and add the operation node
                    node = OperationNode(func_name, inputs, output_var, attributes)
                    graph.add_node(node)
            
            elif isinstance(call_node.func.value, ast.Attribute):
                # Method call on an attribute (e.g., x.view(x.size(0), -1))
                if call_node.func.attr == 'view' or call_node.func.attr == 'reshape':
                    # Handle view/reshape operation
                    base_var = self._extract_value(call_node.func.value)
                    operation_type = call_node.func.attr
                    
                    # Extract shape arguments
                    shape_args = []
                    for arg in call_node.args:
                        shape_args.append(self._extract_value(arg))
                    
                    # Create and add the operation node
                    node = OperationNode(
                        operation_type, 
                        [base_var], 
                        output_var, 
                        {'shape_args': shape_args}
                    )
                    graph.add_node(node)
        # Handle nested calls (e.g., F.relu(self.fc1(x)))
        elif isinstance(call_node.func, ast.Call):
            # This is a nested call, process it recursively
            # First, create a temporary variable for the inner call
            temp_var = f"_temp_{len(graph.nodes)}"
            
            # Process the inner call
            self._process_call_node(call_node.func, temp_var, graph, var_assignments)
            
            # Now process the outer call with the temporary variable as input
            if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
                if call_node.func.value.id == 'F':
                    # Functional call (e.g., F.relu(self.fc1(x)))
                    func_name = call_node.func.attr.lower()
                    attributes = {}
                    
                    # Extract keyword arguments if any
                    for kw in call_node.keywords:
                        attributes[kw.arg] = self._extract_value(kw.value)
                    
                    # Create and add the operation node
                    node = OperationNode(func_name, [temp_var], output_var, attributes)
                    graph.add_node(node)
                elif call_node.func.value.id == 'self':
                    # Module call (e.g., self.fc2(F.relu(self.fc1(x))))
                    module_name = call_node.func.attr
                    
                    # Determine the operation type based on the module name
                    operation_type = module_name.lower()
                    if operation_type.startswith('conv'):
                        if '1d' in module_name:
                            operation_type = 'conv1d'
                        elif '2d' in module_name:
                            operation_type = 'conv2d'
                        elif '3d' in module_name:
                            operation_type = 'conv3d'
                    elif operation_type.startswith('fc') or operation_type.startswith('linear'):
                        operation_type = 'linear'
                    
                    # Create and add the operation node
                    attributes = {'module_attr': module_name}
                    node = OperationNode(operation_type, [temp_var], output_var, attributes)
                    graph.add_node(node)
    
    def _generate_forward_code_from_graph(self, graph: ComputationGraph) -> str:
        """
        Generate Equinox forward code from a computation graph.
        
        Args:
            graph: Computation graph representing the forward pass
            
        Returns:
            String containing Equinox forward code
        """
        lines = []
        
        # Process each node in the graph
        for node in graph.nodes:
            operation_type = node.operation_type
            
            if operation_type in self.operation_converters:
                # Use the registered converter for this operation type
                converter_func = self.operation_converters[operation_type]
                line = converter_func(node)
                lines.append(line)
            else:
                # Default handling for unknown operations
                inputs_str = ", ".join(node.inputs)
                lines.append(f"{node.output} = {operation_type}({inputs_str})")
        
        # Add return statement if output variable is set
        if graph.output_var:
            lines.append(f"return {graph.output_var}")
        else:
            # Default to returning the output of the last operation
            if graph.nodes:
                lines.append(f"return {graph.nodes[-1].output}")
        
        return "\n".join(lines)
    
    def _convert_linear_op(self, node: OperationNode) -> str:
        """Convert a linear operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            # Use the actual module name (fc1, fc2, etc.) instead of the operation type
            module_attr = node.attributes.get('module_attr', 'fc1')
            module_name = f"self.{module_attr}"
            return f"{node.output} = jax.vmap({module_name})({input_var})"
        return ""
    
    def _convert_conv1d_op(self, node: OperationNode) -> str:
        """Convert a conv1d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            # Use the actual module attribute name (conv1, conv2, etc.) instead of the generic type
            module_attr = node.attributes.get('module_attr', 'conv1')
            module_name = f"self.{module_attr}"
            return f"{node.output} = jax.vmap({module_name})({input_var})"
        return ""
    
    def _convert_conv2d_op(self, node: OperationNode) -> str:
        """Convert a conv2d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            # Use the actual module attribute name (conv1, conv2, etc.) instead of the generic type
            module_attr = node.attributes.get('module_attr', 'conv1')
            module_name = f"self.{module_attr}"
            return f"{node.output} = jax.vmap({module_name})({input_var})"
        return ""
    
    def _convert_conv3d_op(self, node: OperationNode) -> str:
        """Convert a conv3d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            # Use the actual module attribute name (conv1, conv2, etc.) instead of the generic type
            module_attr = node.attributes.get('module_attr', 'conv1')
            module_name = f"self.{module_attr}"
            return f"{node.output} = jax.vmap({module_name})({input_var})"
        return ""
    
    def _convert_relu_op(self, node: OperationNode) -> str:
        """Convert a ReLU operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            return f"{node.output} = jax.nn.relu({input_var})"
        return ""
    
    def _convert_max_pool1d_op(self, node: OperationNode) -> str:
        """Convert a max_pool1d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            kernel_size = node.attributes.get('kernel_size', 2)
            strides = node.attributes.get('stride', kernel_size)
            padding = node.attributes.get('padding', 0)
            return f"{node.output} = jax.nn.max_pool({input_var}, window_shape=({kernel_size},), strides=({strides},), padding='{padding}')"
        return ""
    
    def _convert_max_pool2d_op(self, node: OperationNode) -> str:
        """Convert a max_pool2d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            kernel_size = node.attributes.get('kernel_size', 2)
            strides = node.attributes.get('stride', kernel_size)
            padding = node.attributes.get('padding', 0)
            
            # Handle different kernel_size formats
            if isinstance(kernel_size, (list, tuple)):
                kernel_shape = f"({kernel_size[0]}, {kernel_size[1]})"
            else:
                kernel_shape = f"({kernel_size}, {kernel_size})"
                
            # Handle different strides formats
            if isinstance(strides, (list, tuple)):
                strides_shape = f"({strides[0]}, {strides[1]})"
            else:
                strides_shape = f"({strides}, {strides})"
            
            # Convert padding to the format expected by JAX
            if padding == 'same':
                padding_str = "'SAME'"
            elif padding == 'valid':
                padding_str = "'VALID'"
            else:
                # For integer padding, use the JAX padding format
                padding_str = f"'{padding}'"
                
            return f"{node.output} = jax.nn.max_pool({input_var}, window_shape={kernel_shape}, strides={strides_shape}, padding={padding_str})"
        return ""
    
    def _convert_max_pool3d_op(self, node: OperationNode) -> str:
        """Convert a max_pool3d operation to Equinox code."""
        if len(node.inputs) > 0:
            input_var = node.inputs[0]
            kernel_size = node.attributes.get('kernel_size', 2)
            strides = node.attributes.get('stride', kernel_size)
            padding = node.attributes.get('padding', 0)
            
            # Handle different kernel_size formats
            if isinstance(kernel_size, (list, tuple)):
                kernel_shape = f"({kernel_size[0]}, {kernel_size[1]}, {kernel_size[2]})"
            else:
                kernel_shape = f"({kernel_size}, {kernel_size}, {kernel_size})"
                
            # Handle different strides formats
            if isinstance(strides, (list, tuple)):
                strides_shape = f"({strides[0]}, {strides[1]}, {strides[2]})"
            else:
                strides_shape = f"({strides}, {strides}, {strides})"
                
            return f"{node.output} = jax.nn.max_pool({input_var}, window_shape={kernel_shape}, strides={strides_shape}, padding='{padding}')"
        return ""
    
    def _convert_view_op(self, node: OperationNode) -> str:
        """Convert a view operation to Equinox code."""
        if len(node.inputs) > 0 and 'shape_args' in node.attributes:
            input_var = node.inputs[0]
            shape_args = node.attributes['shape_args']
            
            # Handle special case for flattening (e.g., x.view(x.size(0), -1))
            if len(shape_args) == 2 and shape_args[1] == -1:
                # For batch flattening, preserve the batch dimension
                if 'size(0)' in str(shape_args[0]) or shape_args[0] == 'x.size(0)':
                    return f"{node.output} = jnp.reshape({input_var}, ({input_var}.shape[0], -1))"
                else:
                    return f"{node.output} = jnp.reshape({input_var}, ({shape_args[0]}, -1))"
            else:
                # Handle dynamic shapes that might reference tensor dimensions
                processed_shape_args = []
                for arg in shape_args:
                    arg_str = str(arg)
                    if 'size(' in arg_str:
                        # Extract the dimension index from size(index)
                        dim_match = re.search(r'size\((\d+)\)', arg_str)
                        if dim_match:
                            dim_index = dim_match.group(1)
                            processed_shape_args.append(f"{input_var}.shape[{dim_index}]")
                        else:
                            processed_shape_args.append(arg_str)
                    else:
                        processed_shape_args.append(arg_str)
                
                shape_str = ", ".join(processed_shape_args)
                return f"{node.output} = jnp.reshape({input_var}, ({shape_str}))"
        return ""
    
    def _convert_reshape_op(self, node: OperationNode) -> str:
        """Convert a reshape operation to Equinox code."""
        # Reshape is essentially the same as view in JAX
        return self._convert_view_op(node)
    
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
        
        # Add the forward code with proper indentation
        indent = "        "
        for line in forward_code.split('\n'):
            if line.strip():
                lines.append(indent + line)
        
        return '\n'.join(lines)
