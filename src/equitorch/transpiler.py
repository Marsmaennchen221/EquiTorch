"""Complete transpiler using proper AST transformation patterns and modular converters.

Replaces the complex manual AST processing with clean, standard patterns
and uses separated converter modules for better organization.
"""
import ast

from .converters import (
    # Functional converters
    get_method_name,
    get_module_converter,
    # Module converters
    is_supported_module,
    is_tensor_method_call,
    transform_functional_call,
    # Method converters
    transform_method_call,
)


class PyTorchToEquinoxTransformer(ast.NodeTransformer):
    """Complete transpiler using proper AST transformation patterns and modular converters.

    This replaces the complex manual AST processing with clean, standard patterns
    and uses separated converter modules for better organization.
    """

    def __init__(self):
        """Initialize the transpiler."""
        self.current_class_params = []  # Track parameters for current class
        self.key_counter = 0
        self.required_class_definitions = set()  # Track classes that need to be injected
    
    def _filter_pytorch_imports(self, body: list[ast.stmt]) -> list[ast.stmt]:
        """Filter out PyTorch imports from the module body.
        
        Args:
            body: List of AST statements from module body
            
        Returns:
            Filtered list without PyTorch imports
        """
        filtered_body = []
        pytorch_modules = {'torch', 'torch.nn', 'torch.nn.functional'}
        
        for stmt in body:
            if isinstance(stmt, ast.Import):
                # Filter out PyTorch imports
                filtered_names = []
                for alias in stmt.names:
                    if not any(alias.name.startswith(pytorch_mod) for pytorch_mod in pytorch_modules):
                        filtered_names.append(alias)
                if filtered_names:
                    stmt.names = filtered_names
                    filtered_body.append(stmt)
            elif isinstance(stmt, ast.ImportFrom):
                # Filter out PyTorch from imports
                if stmt.module and not any(stmt.module.startswith(pytorch_mod) for pytorch_mod in pytorch_modules):
                    filtered_body.append(stmt)
            else:
                filtered_body.append(stmt)
        
        return filtered_body
    
    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        """Transform the entire module."""
        # Add JAX/Equinox imports at the top
        imports = [
            ast.Import(names=[ast.alias(name="jax", asname=None)]),
            ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]),
            ast.Import(names=[ast.alias(name="equinox", asname="eqx")]),
        ]
        
        # Transform the rest of the module
        new_node: ast.Module = self.generic_visit(node)  # type: ignore[assignment]
        
        # Filter out PyTorch imports
        filtered_body = self._filter_pytorch_imports(new_node.body)
        
        # Inject required class definitions
        class_definitions = self._get_required_class_definitions()
        
        # Prepend imports and append class definitions
        new_node.body = imports + filtered_body + class_definitions
        
        return new_node
    
    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:  # noqa: N802
        """Transform PyTorch module classes to Equinox."""
        # Check if this is a PyTorch module
        if not self._is_pytorch_module(node):
            return node
        
        # Reset state for this class
        self.current_class_params = []
        self.key_counter = 0
        
        # Extract parameters from __init__ method
        init_method = self._find_init_method(node)
        if init_method:
            self._extract_parameters(init_method)
        
        # Transform the class
        new_node: ast.ClassDef = self.generic_visit(node)  # type: ignore[assignment]
        
        # Change base class to eqx.Module
        new_node.bases = [ast.Attribute(
            value=ast.Name(id="eqx", ctx=ast.Load()),
            attr="Module",
            ctx=ast.Load(),
        )]
        
        # Add type annotations for parameters
        self._add_type_annotations(new_node)
        
        return new_node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        """Transform methods."""
        if node.name == "__init__":
            return self._transform_init_method(node)
        if node.name == "forward":
            return self._transform_forward_method(node)
        return node
    
    def visit_Assign(self, node: ast.Assign) -> list[ast.stmt]:  # noqa: N802
        """Transform assignments, adding key splitting for module initializations."""
        # First, recursively transform the assignment
        node = self.generic_visit(node)  # type: ignore[assignment]
        
        # Check if this assignment needs key splitting (contains subkey usage)
        if self._assignment_needs_key_splitting(node):
            # This assignment uses subkey - add key splitting
            statements = []
            
            # Add key splitting: key, subkey = jax.random.split(key)
            key_split = ast.Assign(
                targets=[ast.Tuple(
                    elts=[
                        ast.Name(id="key", ctx=ast.Store()),
                        ast.Name(id="subkey", ctx=ast.Store())
                    ],
                    ctx=ast.Store()
                )],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(
                            value=ast.Name(id="jax", ctx=ast.Load()),
                            attr="random",
                            ctx=ast.Load()
                        ),
                        attr="split",
                        ctx=ast.Load()
                    ),
                    args=[ast.Name(id="key", ctx=ast.Load())],
                    keywords=[]
                )
            )
            statements.append(key_split)
            
            # Add the original assignment (which now has subkey in the call)
            statements.append(node)
            
            return statements
        
        return [node]
    
    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802
        """Using visitor pattern here now. nested calls are just handled recursively."""
        # First, recursively transform any nested calls
        node = self.generic_visit(node)  # type: ignore[assignment]
        
        # Now transform this specific call
        if isinstance(node.func, ast.Attribute):
            if self._is_self_module_call(node):
                return self._transform_module_call(node)
            if self._is_functional_call(node):
                return self._transform_functional_call(node)
            if is_tensor_method_call(node):
                return self._transform_method_call(node)
            if self._is_pytorch_module_init(node):
                return self._transform_module_init(node)
        elif isinstance(node.func, ast.Name):
            # Handle custom class instantiations
            if self._is_custom_class_init(node):
                return self._transform_custom_class_init(node)
        
        return node
    
    def _is_pytorch_module(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from nn.Module."""
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                return True
            if isinstance(base, ast.Name) and base.id == "Module":
                return True
        return False
    
    def _find_init_method(self, node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find the __init__ method."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return item
        return None
    
    def _extract_parameters(self, init_method: ast.FunctionDef):
        """Extract ALL self.param = Class(...) assignments for type annotations."""
        class ParameterExtractor(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
            
            def visit_Assign(self, node):  # noqa: N802
                if (len(node.targets) == 1 and
                    isinstance(node.targets[0], ast.Attribute) and
                    isinstance(node.targets[0].value, ast.Name) and
                    node.targets[0].value.id == "self" and
                    isinstance(node.value, ast.Call)):
                    
                    param_name = node.targets[0].attr
                    
                    # Handle both PyTorch modules and custom classes
                    if self.parent._is_pytorch_module_init(node.value):
                        # PyTorch module - use existing logic
                        module_type = node.value.func.attr
                        self.parent.current_class_params.append({
                            "name": param_name,
                            "type": module_type,
                            "call_node": node.value,
                            "transformed_node": None,  # Will be filled during transformation
                            "type_annotation": None,   # Will be filled during transformation
                        })
                    elif isinstance(node.value.func, ast.Name):
                        # Custom class instantiation - track for type annotations
                        class_name = node.value.func.id
                        self.parent.current_class_params.append({
                            "name": param_name,
                            "type": class_name,
                            "call_node": node.value,
                            "transformed_node": None,
                            "type_annotation": ast.Name(id=class_name, ctx=ast.Load()),
                        })
        
        extractor = ParameterExtractor(self)
        extractor.visit(init_method)
    
    def _add_type_annotations(self, class_node: ast.ClassDef):
        """Add type annotations for parameters using type info from converters."""
        annotations = []
        for param in self.current_class_params:
            # Use the type annotation provided by the converter if available
            if param["type_annotation"] is not None:
                annotation_node = param["type_annotation"]
            else:
                # Fallback to auto-transpiled module name for custom modules
                annotation_node = ast.Name(id=param["type"], ctx=ast.Load())
            
            annotation = ast.AnnAssign(
                target=ast.Name(id=param["name"], ctx=ast.Store()),
                annotation=annotation_node,
                value=None,
                simple=1,
            )
            annotations.append(annotation)
        
        # Insert annotations at the beginning of class body
        class_node.body = annotations + class_node.body
    
    def _transform_init_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform __init__ method."""
        # Add key parameter with default None
        if not any(arg.arg == "key" for arg in node.args.args):
            node.args.args.append(ast.arg(arg="key", annotation=None))
            node.args.defaults.append(ast.Constant(value=None))
        
        # Add **kwargs if not present
        if not node.args.kwarg:
            node.args.kwarg = ast.arg(arg="kwargs", annotation=None)
        
        # Transform the body
        new_body: list[ast.stmt] = []
        
        # Add key initialization
        key_init = ast.If(
            test=ast.Compare(
                left=ast.Name(id="key", ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id="key", ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id="jax", ctx=ast.Load()),
                                attr="random",
                                ctx=ast.Load(),
                            ),
                            attr="PRNGKey",
                            ctx=ast.Load(),
                        ),
                        args=[ast.Constant(value=0)],
                        keywords=[],
                    ),
                ),
            ],
            orelse=[],
        )
        new_body.append(key_init)
        
        # Transform existing body, filtering out any statements containing super() calls
        for stmt in node.body:
            if self._contains_super_call(stmt):
                continue  # Skip any statement that contains a super() call
            
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        
        node.body = new_body
        return node
    
    def _transform_forward_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform forward method to __call__."""
        node.name = "__call__"
        
        # Transform the body - this automatically handles all nested expressions!
        return self.generic_visit(node)  # type: ignore[return-value]
    
    def _is_self_module_call(self, node: ast.Call) -> bool:
        """Check if this is a self.module(x) call."""
        return (isinstance(node.func.value, ast.Name) and  # type: ignore[attr-defined]
                node.func.value.id == "self")  # type: ignore[attr-defined]
    
    def _is_functional_call(self, node: ast.Call) -> bool:
        """Check if this is a F.function(x) or torch.function(x) call."""
        return (isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id in ["F", "torch"])  # type: ignore[attr-defined]
    
    def _is_pytorch_module_init(self, node: ast.Call) -> bool:
        """Check if this is a PyTorch module initialization."""
        if isinstance(node.func, ast.Attribute):
            module_name = node.func.attr
            return is_supported_module(module_name)
        return False
    
    def _transform_module_call(self, node: ast.Call) -> ast.Call:
        """Transform self.module(x) -> direct call or jax.vmap(self.module)(x) based on module type."""
        module_name = node.func.attr  # type: ignore[attr-defined]
        
        # Check if this is an RNN call that needs special handling
        if self._is_rnn_module_call(module_name):
            return self._transform_rnn_call(node, module_name)
        
        # Check if this module handles batches natively (Conv2d, Linear, etc.)
        if self._module_handles_batches_natively(module_name):
            # For modules that handle batches natively, just call them directly
            return node
        
        # For other modules, use vmap
        vmap_func = ast.Attribute(
            value=ast.Name(id="jax", ctx=ast.Load()),
            attr="vmap",
            ctx=ast.Load(),
        )
        
        module_ref = ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=module_name,
            ctx=ast.Load(),
        )
        
        vmap_call = ast.Call(
            func=vmap_func,
            args=[module_ref],
            keywords=[],
        )
        
        return ast.Call(
            func=vmap_call,
            args=node.args,
            keywords=node.keywords,
        )
    
    def _transform_functional_call(self, node: ast.Call) -> ast.Call:
        """Transform F.function(x) -> jax.nn.function(x) using modular converters."""
        module_name = node.func.value.id  # type: ignore[attr-defined]
        func_name = node.func.attr  # type: ignore[attr-defined]
        
        # Use the modular functional converter
        return transform_functional_call(node, module_name, func_name)
    
    def _transform_method_call(self, node: ast.Call) -> ast.Call:
        """Transform x.method() calls using modular converters."""
        method_name = get_method_name(node)
        if method_name:
            return transform_method_call(node, method_name)
        return node
    
    def _is_rnn_module_call(self, module_name: str) -> bool:
        """Check if this is a call to an RNN-like module that needs scan."""
        # Extended RNN type mapping
        RNN_MODULE_TYPES = {"RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell"}
        
        # Check if the module is an RNN type based on our parameter tracking
        for param in self.current_class_params:
            if param["name"] == module_name and param["type"] in RNN_MODULE_TYPES:
                return True
        return False
    
    def _module_handles_batches_natively(self, module_name: str) -> bool:
        """Check if a module handles batches natively and doesn't need vmap."""
        # Get the module type for this parameter
        module_type = None
        for param in self.current_class_params:
            if param["name"] == module_name:
                module_type = param["type"]
                break
        
        if module_type is None:
            return False
        
        # In Equinox, ALL modules operate on single examples, and batching
        # is handled with vmap at the top level (not inside the model)
        # So we should NOT use vmap for any modules inside the model
        
        # Built-in PyTorch modules that are converted to Equinox
        BATCH_NATIVE_MODULES = {
            "Linear", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", 
            "ConvTranspose2d", "ConvTranspose3d", "BatchNorm1d", "BatchNorm2d", 
            "BatchNorm3d", "LayerNorm", "GroupNorm", "Dropout", "Embedding",
            "MultiheadAttention"  # MultiheadAttention handles batches natively
        }
        
        # Check if it's a built-in PyTorch module
        if module_type in BATCH_NATIVE_MODULES:
            return True
        
        # Check if it's a custom class (not a built-in PyTorch module)
        # Custom classes are also designed to work on single examples in Equinox
        # If the module_type is not in our known PyTorch modules, it's likely a custom class
        from .converters.module_converters import MODULE_CONFIGS
        if module_type not in MODULE_CONFIGS:
            # This is a custom class, treat it as batch-native (no vmap needed)
            return True
        
        return False
    
    def _get_rnn_type(self, module_name: str) -> str:
        """Get the RNN type for a given module name."""
        for param in self.current_class_params:
            if param["name"] == module_name:
                return param["type"]
        return "RNN"  # Default fallback
    
    def _create_rnn_scan_function(self, rnn_type: str, module_name: str) -> ast.Lambda:
        """Create proper scan function based on RNN type."""
        
        # Create a single RNN call that we'll reuse
        rnn_call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load()),
                attr=module_name,
                ctx=ast.Load(),
            ),
            args=[
                ast.Name(id="x_t", ctx=ast.Load()),
                ast.Name(id="carry", ctx=ast.Load()),
            ],
            keywords=[],
        )
        
        # Create lambda that returns (new_carry, output)
        # For simple RNNs, both are the same (the new hidden state)
        return ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="carry", annotation=None),
                    ast.arg(arg="x_t", annotation=None),
                ],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            body=ast.Tuple(
                elts=[rnn_call, rnn_call],  # (new_carry, output)
                ctx=ast.Load(),
            ),
        )
    
    def _transform_rnn_call(self, node: ast.Call, module_name: str) -> ast.Call:
        """Transform RNN calls to use jax.lax.scan for sequential processing."""
        # For RNN calls like self.rnn(x, hidden), we need to use jax.lax.scan
        # to process the sequence step by step
        
        # Get RNN type for this module
        rnn_type = self._get_rnn_type(module_name)
        
        # Create the scan function
        scan_func = ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="jax", ctx=ast.Load()),
                attr="lax",
                ctx=ast.Load(),
            ),
            attr="scan",
            ctx=ast.Load(),
        )
        
        # Create a proper scan function based on RNN type
        scan_lambda = self._create_rnn_scan_function(rnn_type, module_name)
        
        # Handle input arguments - need to transpose for scan if batch_first=True
        input_seq = node.args[0]
        
        # Create transposed input for scan (batch_first=True means we need seq_first)
        transposed_input = ast.Call(
            func=ast.Attribute(
                value=input_seq,
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                ast.Tuple(
                    elts=[
                        ast.Constant(value=1),  # seq_len
                        ast.Constant(value=0),  # batch_size
                        ast.Constant(value=2),  # features
                    ],
                    ctx=ast.Load(),
                ),
            ],
            keywords=[],
        )
        
        # Handle initial hidden state
        if len(node.args) >= 2:
            init_carry = node.args[1]
            # Remove the first dimension (num_layers=1) and transpose
            init_carry = ast.Subscript(
                value=init_carry,
                slice=ast.Constant(value=0),
                ctx=ast.Load(),
            )
        else:
            # Default hidden state - create zeros with proper shape
            init_carry = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="zeros",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Tuple(
                        elts=[
                            ast.Subscript(
                                value=input_seq,
                                slice=ast.Tuple(
                                    elts=[ast.Constant(value=0)],
                                    ctx=ast.Load(),
                                ),
                                ctx=ast.Load(),
                            ),  # batch_size
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="hidden_size",
                                ctx=ast.Load(),
                            ),
                        ],
                        ctx=ast.Load(),
                    ),
                ],
                keywords=[],
            )
        
        # Create the scan call: final_carry, outputs = jax.lax.scan(scan_fn, init_carry, input_seq)
        scan_call = ast.Call(
            func=scan_func,
            args=[
                scan_lambda,
                init_carry,
                transposed_input,
            ],
            keywords=[],
        )
        
        # Return tuple (outputs, final_carry) to match PyTorch RNN API
        # Note: outputs need to be transposed back to batch_first format
        outputs_transposed = ast.Call(
            func=ast.Attribute(
                value=ast.Subscript(
                    value=scan_call,
                    slice=ast.Constant(value=1),
                    ctx=ast.Load(),
                ),
                attr="transpose",
                ctx=ast.Load(),
            ),
            args=[
                ast.Tuple(
                    elts=[
                        ast.Constant(value=1),  # batch_size
                        ast.Constant(value=0),  # seq_len
                        ast.Constant(value=2),  # features
                    ],
                    ctx=ast.Load(),
                ),
            ],
            keywords=[],
        )
        
        return ast.Tuple(
            elts=[
                outputs_transposed,
                ast.Subscript(
                    value=scan_call,
                    slice=ast.Constant(value=0),
                    ctx=ast.Load(),
                ),  # final_carry
            ],
            ctx=ast.Load(),
        )
    
    def _store_type_annotation(self, original_node: ast.Call, type_annotation: ast.AST):
        """Store type annotation for a parameter based on its original call node."""
        # Find the parameter that corresponds to this call node
        for param in self.current_class_params:
            if param["call_node"] is original_node:
                param["type_annotation"] = type_annotation
                break
    
    def _contains_super_call(self, node: ast.AST) -> bool:
        """Recursively check if an AST node contains a super() call anywhere."""
        class SuperCallFinder(ast.NodeVisitor):
            def __init__(self):
                self.found_super = False
            
            def visit_Call(self, node):
                if (isinstance(node.func, ast.Name) and node.func.id == "super"):
                    self.found_super = True
                self.generic_visit(node)  # Continue searching in child nodes
        
        finder = SuperCallFinder()
        finder.visit(node)
        return finder.found_super
    
    def _assignment_needs_key_splitting(self, node: ast.Assign) -> bool:
        """Check if an assignment contains subkey usage and needs key splitting."""
        class SubkeyFinder(ast.NodeVisitor):
            def __init__(self):
                self.found_subkey = False
            
            def visit_Name(self, node):
                if node.id == "subkey":
                    self.found_subkey = True
        
        finder = SubkeyFinder()
        finder.visit(node)
        return finder.found_subkey
    
    def _is_custom_class_init(self, node: ast.Call) -> bool:
        """Check if this is a custom class instantiation that needs a key."""
        if isinstance(node.func, ast.Name):
            # Check if this class name is in our tracked parameters
            class_name = node.func.id
            for param in self.current_class_params:
                if param["type"] == class_name:
                    return True
        return False
    
    def _transform_custom_class_init(self, node: ast.Call) -> ast.Call:
        """Transform custom class instantiation to include key parameter."""
        # Add key parameter to custom class calls
        self._add_key_to_call(node)
        return node
    
    def _add_key_to_call(self, call_node: ast.Call):
        """Add key parameter to a module call using functional key splitting."""
        # Check if key parameter already exists
        has_key = any(kw.arg == "key" for kw in call_node.keywords if kw.arg is not None)
        
        if not has_key:
            # Add the subkey to the call - the key splitting will be handled in visit_Assign
            key_keyword = ast.keyword(
                arg="key",
                value=ast.Name(id="subkey", ctx=ast.Load())
            )
            call_node.keywords.append(key_keyword)
    
    def _transform_module_init(self, node: ast.Call) -> ast.Call:
        """Transform PyTorch module initialization to Equinox using modular converters."""
        module_type = node.func.attr  # type: ignore[attr-defined]
        if is_supported_module(module_type):
            converter = get_module_converter(module_type)
            
            # Track if this module needs a class definition to be injected
            self.required_class_definitions.add(module_type)
            
            # Call converter and handle both old and new interface
            result = converter(node, 0)  # Pass 0 as key_counter since we'll use functional splitting
            
            # Check if converter returns tuple (new interface) or just node (old interface)
            if isinstance(result, tuple):
                transformed_node, type_annotation = result
                # Store the type annotation for this parameter
                self._store_type_annotation(node, type_annotation)
                # Don't add key parameter for Parameter - it's handled by functional converter
                if module_type != "Parameter":
                    self._add_key_to_call(transformed_node)
                return transformed_node
            else:
                # Old interface - add key parameter (except for Parameter)
                if module_type != "Parameter":
                    self._add_key_to_call(result)
                return result
        return node
    
    def _get_required_class_definitions(self) -> list[ast.stmt]:
        """Generate AST nodes for required class definitions using the unified system."""
        from .converters.module_converters import (
            get_required_custom_implementations,
            auto_transpile_module,
            MODULE_CONFIGS,
        )
        
        class_definitions = []
        
        # Get custom implementations for PyTorch modules that need them
        custom_implementations = get_required_custom_implementations(self.required_class_definitions)
        
        for custom_name, definition in custom_implementations.items():
            class_code = definition.strip()
            class_ast = ast.parse(class_code)
            class_definitions.extend(class_ast.body)
        
        # Handle auto-transpiled modules (those not in MODULE_CONFIGS)
        for pytorch_module in self.required_class_definitions:
            if pytorch_module not in MODULE_CONFIGS:
                # This is an auto-transpiled class (not in our configured modules)
                try:
                    class_def_str, _ = auto_transpile_module(pytorch_module)
                    class_ast = ast.parse(class_def_str)
                    class_definitions.extend(class_ast.body)
                except Exception:
                    # If auto-transpilation fails, skip this class
                    pass
        
        return class_definitions


def transpile(pytorch_code: str) -> str:
    """Complete transpiler using proper AST transformation patterns and modular converters.
    
    Args:
        pytorch_code: PyTorch module code
        
    Returns:
        Equivalent Equinox code
    """
    # Parse the code
    tree = ast.parse(pytorch_code)
    
    # Transform it
    transformer = PyTorchToEquinoxTransformer()
    new_tree = transformer.visit(tree)
    
    # Fix missing locations
    ast.fix_missing_locations(new_tree)
    
    # Convert back to code
    return ast.unparse(new_tree)
