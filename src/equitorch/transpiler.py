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
    
    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        """Transform the entire module."""
        # Add imports at the top
        imports = [
            ast.Import(names=[ast.alias(name="jax", asname=None)]),
            ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]),
            ast.Import(names=[ast.alias(name="equinox", asname="eqx")]),
        ]
        
        # Transform the rest of the module
        new_node: ast.Module = self.generic_visit(node)  # type: ignore[assignment]
        
        # Inject required class definitions
        class_definitions = self._get_required_class_definitions()
        
        # Prepend imports and append class definitions
        new_node.body = imports + new_node.body + class_definitions
        
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
        
        # Check if this is a self.param = Module(...) assignment
        if (len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Attribute) and
            isinstance(node.targets[0].value, ast.Name) and
            node.targets[0].value.id == "self" and
            isinstance(node.value, ast.Call) and
            self._is_pytorch_module_init(node.value)):
            
            # This is a module assignment - add key splitting
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
        """Extract module parameters from __init__ method."""
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
                    if self.parent._is_pytorch_module_init(node.value):
                        module_type = node.value.func.attr
                        # We'll fill in transformed_node and type_annotation later during transformation
                        self.parent.current_class_params.append({
                            "name": param_name,
                            "type": module_type,
                            "call_node": node.value,
                            "transformed_node": None,  # Will be filled during transformation
                            "type_annotation": None,   # Will be filled during transformation
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
        
        # Transform existing body
        for stmt in node.body:
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
        return (isinstance(node.func.value, ast.Name) and  # type: ignore[attr-defined]
                node.func.value.id in ["F", "torch"])  # type: ignore[attr-defined]
    
    def _is_pytorch_module_init(self, node: ast.Call) -> bool:
        """Check if this is a PyTorch module initialization."""
        if isinstance(node.func, ast.Attribute):
            module_name = node.func.attr
            return is_supported_module(module_name)
        return False
    
    def _transform_module_call(self, node: ast.Call) -> ast.Call:
        """Transform self.module(x) -> jax.vmap(self.module)(x) or special handling for RNNs."""
        module_name = node.func.attr  # type: ignore[attr-defined]
        
        # Check if this is an RNN call that needs special handling
        if self._is_rnn_module_call(module_name):
            return self._transform_rnn_call(node, module_name)
        
        # Create jax.vmap(self.module)(args)
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
        # Check if the module is an RNN type based on our parameter tracking
        for param in self.current_class_params:
            if param["name"] == module_name and param["type"] == "RNN":
                return True
        return False
    
    def _transform_rnn_call(self, node: ast.Call, module_name: str) -> ast.Call:
        """Transform RNN calls to use jax.lax.scan for sequential processing."""
        # For RNN calls like self.rnn(x, hidden), we need to use jax.lax.scan
        # to process the sequence step by step
        
        # Create a scan function that processes one timestep
        # def scan_fn(carry, x_t):
        #     hidden = carry
        #     new_hidden = self.rnn(x_t, hidden)
        #     return new_hidden, new_hidden
        
        # For now, let's create a simplified version that uses jax.lax.scan
        # This is a placeholder - the actual implementation would be more complex
        scan_func = ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="jax", ctx=ast.Load()),
                attr="lax",
                ctx=ast.Load(),
            ),
            attr="scan",
            ctx=ast.Load(),
        )
        
        # Create a lambda function for the scan operation
        # lambda carry, x_t: (self.rnn(x_t, carry), self.rnn(x_t, carry))
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
        
        scan_lambda = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[
                    ast.arg(arg="carry", annotation=None),
                    ast.arg(arg="x_t", annotation=None),
                ],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=ast.Tuple(
                elts=[rnn_call, rnn_call],
                ctx=ast.Load(),
            ),
        )
        
        # Create the scan call: jax.lax.scan(scan_fn, init_hidden, x)
        # We'll use the second argument as initial hidden state if provided
        if len(node.args) >= 2:
            init_carry = node.args[1]
        else:
            # Default hidden state
            init_carry = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jnp", ctx=ast.Load()),
                    attr="zeros",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Tuple(
                        elts=[
                            ast.Attribute(
                                value=ast.Subscript(
                                    value=node.args[0],
                                    slice=ast.Constant(value=0),
                                    ctx=ast.Load(),
                                ),
                                attr="shape",
                                ctx=ast.Load(),
                            ),
                            ast.Attribute(
                                value=ast.Name(id="self", ctx=ast.Load()),
                                attr="hidden_size",
                                ctx=ast.Load(),
                            ),
                        ],
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            )
        
        return ast.Call(
            func=scan_func,
            args=[
                scan_lambda,
                init_carry,
                node.args[0],  # input sequence
            ],
            keywords=[],
        )
    
    def _store_type_annotation(self, original_node: ast.Call, type_annotation: ast.AST):
        """Store type annotation for a parameter based on its original call node."""
        # Find the parameter that corresponds to this call node
        for param in self.current_class_params:
            if param["call_node"] is original_node:
                param["type_annotation"] = type_annotation
                break
    
    def _add_key_to_call(self, call_node: ast.Call):
        """Add key parameter to a module call using functional key splitting."""
        # Check if key parameter already exists
        has_key = any(kw.arg == "key" for kw in call_node.keywords if kw.arg is not None)
        
        if not has_key:
            # Create functional key splitting: key, subkey = jax.random.split(key)
            # Add the subkey to the call
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
                # Add key parameter to the transformed node
                self._add_key_to_call(transformed_node)
                return transformed_node
            else:
                # Old interface - add key parameter
                self._add_key_to_call(result)
                return result
        return node
    
    def _get_required_class_definitions(self) -> list[ast.stmt]:
        """Generate AST nodes for required class definitions using the generic system."""
        from .converters.module_converters import (
            get_required_custom_implementations,
            auto_transpile_module,
            PYTORCH_TO_CUSTOM_MAPPING,
        )
        
        class_definitions = []
        
        # Get custom implementations for PyTorch modules that need them
        custom_implementations = get_required_custom_implementations(self.required_class_definitions)
        
        for custom_name, definition in custom_implementations.items():
            class_code = definition.strip()
            class_ast = ast.parse(class_code)
            class_definitions.extend(class_ast.body)
        
        # Handle auto-transpiled modules (those not in custom mappings and not in manual mappings)
        from .converters.module_converters import MODULE_MAPPINGS
        
        for pytorch_module in self.required_class_definitions:
            if (pytorch_module not in PYTORCH_TO_CUSTOM_MAPPING and 
                pytorch_module not in MODULE_MAPPINGS):
                # This is an auto-transpiled class (not custom, not manual)
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
