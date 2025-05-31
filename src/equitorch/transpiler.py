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
                        self.parent.current_class_params.append({
                            "name": param_name,
                            "type": module_type,
                            "call_node": node.value,
                        })
        
        extractor = ParameterExtractor(self)
        extractor.visit(init_method)
    
    def _add_type_annotations(self, class_node: ast.ClassDef):
        """Add type annotations for parameters using correct transpiled types."""
        from .converters.module_converters import PYTORCH_TO_CUSTOM_MAPPING
        
        annotations = []
        for param in self.current_class_params:
            pytorch_type = param["type"]
            
            # Determine the correct type annotation
            if pytorch_type in PYTORCH_TO_CUSTOM_MAPPING:
                # Use custom implementation name
                custom_type = PYTORCH_TO_CUSTOM_MAPPING[pytorch_type]
                annotation_node = ast.Name(id=custom_type, ctx=ast.Load())
            elif pytorch_type in ["Linear", "Conv1d", "Conv2d", "Conv3d", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "Dropout", "LSTM"]:
                # Use eqx.nn.ModuleName for standard Equinox modules
                if pytorch_type == "LSTM":
                    # LSTM maps to LSTMCell
                    equinox_type = "LSTMCell"
                else:
                    equinox_type = pytorch_type
                    
                annotation_node = ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id="eqx", ctx=ast.Load()),
                        attr="nn",
                        ctx=ast.Load(),
                    ),
                    attr=equinox_type,
                    ctx=ast.Load(),
                )
            else:
                # Auto-transpiled module - use the module name directly
                annotation_node = ast.Name(id=pytorch_type, ctx=ast.Load())
            
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
            new_body.append(self.visit(stmt))
        
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
        """Transform self.module(x) -> jax.vmap(self.module)(x)."""
        module_name = node.func.attr  # type: ignore[attr-defined]
        
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
    
    def _transform_module_init(self, node: ast.Call) -> ast.Call:
        """Transform PyTorch module initialization to Equinox using modular converters."""
        module_type = node.func.attr  # type: ignore[attr-defined]
        if is_supported_module(module_type):
            converter = get_module_converter(module_type)
            self.key_counter += 1
            
            # Track if this module needs a class definition to be injected
            self.required_class_definitions.add(module_type)
            
            return converter(node, self.key_counter - 1)
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
