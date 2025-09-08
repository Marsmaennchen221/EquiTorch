"""Complete transpiler using proper AST transformation patterns and modular converters.

Replaces the complex manual AST processing with clean, standard patterns
and uses separated converter modules for better organization.
"""
import ast

from equitorch.converters import (
    # Functional converters
    get_method_name,
    get_module_converter,
    # Module converters
    is_supported_module,
    is_tensor_method_call,
)


class PyTorchToEquinoxTransformer(ast.NodeTransformer):
    """Complete transpiler using proper AST transformation patterns and modular converters.

    This replaces the complex manual AST processing with clean, standard patterns
    and uses separated converter modules for better organization.
    """

    def __init__(self):
        """Initialize the transpiler."""
        self.key_counter = 0
        self.current_class_params: dict[str, dict[str, str | ast.Call]] = {}
        self.current_imports = {}
        self.required_class_definitions = set()  # Track classes that need to be injected


    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:  # noqa: N802
        """Save from Imports."""
        node_module: str = node.module if node.module is not None else ""
        for alias in node.names:
            if alias.asname is not None:
                self.current_imports[alias.asname] = node_module + "." + alias.name
            else:
                self.current_imports[alias.name] = node_module + "." + alias.name

        new_node: ast.ImportFrom = self.generic_visit(node)  # type: ignore[assignment]
        return new_node

    def visit_Import(self, node: ast.Import) -> ast.Import:  # noqa: N802
        """Save regular Imports."""
        for alias in node.names:
            if alias.asname is not None:
                self.current_imports[alias.asname] = alias.name
            else:
                self.current_imports[alias.name.split(".")[-1]] = alias.name

        new_node: ast.Import = self.generic_visit(node)  # type: ignore[assignment]
        return new_node

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
        for stmt in node.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self.visit(stmt)
        
        # Add JAX/Equinox imports at the top
        imports = [
            ast.Import(names=[ast.alias(name="jax", asname=None)]),
            ast.Import(names=[ast.alias(name="jax.numpy", asname="jnp")]),
            ast.Import(names=[ast.alias(name="jax.random", asname="jr")]),
            ast.Import(names=[ast.alias(name="equinox", asname="eqx")]),
            ast.Import(names=[ast.alias(name="equitorch.lib", asname="eqt")])
        ]

        original_imports = self.current_imports.copy()

        new_node: ast.Module = self.generic_visit(node)  # type: ignore[assignment]

        self.current_imports = original_imports

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

        # Track parameters for current class
        self.current_class_params: dict[str, dict[str, str | ast.Call]] = {}
        self.key_counter = 0

        # Extract parameters from __init__ method
        init_method = self._find_init_method(node)
        if init_method is not None:
            self.current_class_params.update(self._extract_parameters(init_method))

        # Transform the class
        new_node: ast.ClassDef = self.generic_visit(node)  # type: ignore[assignment]

        # Change base class to eqx.Module
        base_index = self._get_pytorch_module_base_index(node)
        new_node.bases[base_index] = ast.Attribute(
            value=ast.Name(id="eqx", ctx=ast.Load()),
            attr="Module",
            ctx=ast.Load(),
        )

        # Add type annotations for parameters if parameters exist
        if self.current_class_params is not None:
            self._add_type_annotations(new_node, self.current_class_params)

        return new_node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        """Transform methods."""
        if node.name == "__init__":
            return self._transform_init_method(node)
        if node.name == "forward":
            return self._transform_forward_method(node)
        
        return self.generic_visit(node)  # type: ignore[return-value]

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
                        ast.Name(id=f"subkey", ctx=ast.Store())
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
            self.key_counter += 1

            return statements

        return [node]

    def visit_Call(self, node: ast.Call) -> ast.Call | ast.Tuple:  # noqa: N802
        """Using visitor pattern here now. nested calls are just handled recursively."""
        # First, recursively transform any nested calls
        node = self.generic_visit(node)  # type: ignore[assignment]

        # Transform keyword arguments that contain PyTorch constructs
        node = self._transform_keyword_arguments(node)

        # Now transform this specific call
        if isinstance(node.func, ast.Attribute):
            if self._is_self_module_call(node):
                return self._transform_module_call(node, self.current_class_params)
            if self._is_functional_call(node):
                return self._transform_functional_call(node)
            if is_tensor_method_call(node):
                return self._transform_method_call(node)
            if self._is_pytorch_module_init(node):
                return self._transform_module_init(node, self.current_class_params)

        elif isinstance(node.func, ast.Name):
            # Handle custom class instantiations
            if self._is_custom_class_init(node, self.current_class_params):
                return self._transform_custom_class_init(node)

        return node

    def _is_pytorch_module(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from torch.nn.Module."""
        for base in node.bases:
            if (
                    (isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Attribute)
                     and base.value.attr == "nn" and isinstance(base.value.value, ast.Name)
                     and self.current_imports[base.value.value.id] == "torch")
                    or (isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name)
                        and self.current_imports[base.value.id] == "torch.nn")
                    or (isinstance(base, ast.Name) and self.current_imports[base.id] == "torch.nn.Module")):
                return True
        return False

    def _get_pytorch_module_base_index(self, node: ast.ClassDef) -> int:
        """Return index of torch.nn.Module base if class inherits it."""
        if self._is_pytorch_module(node):
            for index, base in enumerate(node.bases):
                if (
                        (isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Attribute)
                         and base.value.attr == "nn" and isinstance(base.value.value, ast.Name)
                         and self.current_imports[base.value.value.id] == "torch")
                        or (isinstance(base, ast.Attribute) and base.attr == "Module" and isinstance(base.value, ast.Name)
                            and self.current_imports[base.value.id] == "torch.nn")
                        or (isinstance(base, ast.Name) and self.current_imports[base.id] == "torch.nn.Module")):
                    return index
        raise ValueError("Class does not inherit from torch.nn.Module")

    @staticmethod
    def _find_init_method(node: ast.ClassDef) -> ast.FunctionDef | None:
        """Find the __init__ method."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return item
        return None

    def _extract_parameters(self, init_method: ast.FunctionDef) -> dict[str, dict[str, str | ast.Call]]:
        """Extract ALL self.param = ... assignments for type annotations."""
        class ParameterExtractor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.current_class_params: dict[str, dict[str, str | ast.Call]] = {}

            def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
                if (len(node.targets) == 1 and
                        isinstance(node.targets[0], ast.Attribute) and
                        isinstance(node.targets[0].value, ast.Name) and
                        node.targets[0].value.id == "self"):
                    param_name = node.targets[0].attr
                    
                    if isinstance(node.value, ast.Call):
                        # Handle module calls
                        if PyTorchToEquinoxTransformer._is_pytorch_module_init(node.value):
                            module_type = node.value.func.attr  # type: ignore[attr-defined]
                            self.current_class_params[param_name] = ({
                                "type": module_type,
                                "call_node": node.value,
                                "transformed_node": None,  # Will be filled during transformation
                                "type_annotation": ast.Attribute(
                                    value=ast.Attribute(
                                        value=ast.Name(id="eqx", ctx=ast.Load()),
                                        attr="nn",
                                        ctx=ast.Load(),
                                    ),
                                    attr=str(module_type),
                                    ctx=ast.Load(),
                                )
                            })  # Override same name with latest assignment
                        elif isinstance(node.value.func, ast.Name):
                            class_name = node.value.func.id
                            self.current_class_params[param_name] = ({
                                "type": class_name,
                                "call_node": node.value,
                                "transformed_node": None,
                                "type_annotation": None,  # Will be filled during transformation
                            })
                    else:
                        # Handle non-call assignments (like self.d_model = d_model)
                        # Infer type from the value
                        type_annotation = self._infer_type_annotation(node.value)
                        self.current_class_params[param_name] = ({
                            "type": "attribute",
                            "call_node": None,
                            "transformed_node": None,
                            "type_annotation": type_annotation,
                        })

            def _infer_type_annotation(self, value_node: ast.AST) -> ast.AST:
                """Infer type annotation from assignment value."""
                if isinstance(value_node, ast.Name):
                    # For simple variable assignments, assume int for common cases
                    if value_node.id in ['d_model', 'vocab_size', 'num_heads', 'd_ff', 'hidden_size', 'input_size', 'output_size']:
                        return ast.Name(id="int", ctx=ast.Load())
                elif isinstance(value_node, ast.Constant):
                    if isinstance(value_node.value, int):
                        return ast.Name(id="int", ctx=ast.Load())
                    elif isinstance(value_node.value, float):
                        return ast.Name(id="float", ctx=ast.Load())
                    elif isinstance(value_node.value, str):
                        return ast.Name(id="str", ctx=ast.Load())
                elif isinstance(value_node, ast.Call):
                    # Handle special cases like jax.random.normal -> jnp.ndarray
                    if (isinstance(value_node.func, ast.Attribute) and
                        isinstance(value_node.func.value, ast.Attribute) and
                        isinstance(value_node.func.value.value, ast.Name) and
                        value_node.func.value.value.id == "jax" and
                        value_node.func.value.attr == "random"):
                        return ast.Attribute(
                            value=ast.Name(id="jnp", ctx=ast.Load()),
                            attr="ndarray",
                            ctx=ast.Load()
                        )
                
                # Default fallback
                return ast.Name(id="object", ctx=ast.Load())

        extractor = ParameterExtractor()
        extractor.visit(init_method)
        return extractor.current_class_params

    @staticmethod
    def _add_type_annotations(class_node: ast.ClassDef, current_class_params: dict[str, dict[str, str | ast.Call]]):
        """Add type annotations for parameters using type info from converters."""
        annotations = []
        for name, param_body in current_class_params.items():
            # Use the type annotation provided by the converter if available
            if param_body["type_annotation"] is not None:
                annotation_node = param_body["type_annotation"]
            else:
                # Fallback to auto-transpiled module name for custom modules
                annotation_node = ast.Name(id=param_body["type"], ctx=ast.Load())

            annotation = ast.AnnAssign(
                target=ast.Name(id=name, ctx=ast.Store()),
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
                            value=ast.Name(id="jr", ctx=ast.Load()),
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
        new_body_stmts: list[ast.stmt] = []
        for stmt in node.body:
            if self._contains_super_call(stmt):
                continue  # Skip any statement that contains a super() call

            result = self.visit(stmt)
            if isinstance(result, list):
                new_body_stmts.extend(result)
            else:
                new_body_stmts.append(result)

        # Add key splitting
        key_split = ast.Assign(
            targets=[
                ast.Tuple(
                    elts=[
                        ast.Name(id=f"key_{i}", ctx=ast.Store()) for i in range(self.key_counter)
                    ],
                ),
            ],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="jr", ctx=ast.Load()),
                    attr="split",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Name(id="key", ctx=ast.Load()),
                    ast.Constant(value=self.key_counter),
                ],
                keywords=[],
            ),
        )
        new_body.append(key_split)
        new_body.extend(new_body_stmts)

        node.body = new_body
        return node

    def _transform_forward_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform forward method to __call__."""
        node.name = "__call__"

        # Transform the body - this automatically handles all nested expressions!
        return self.generic_visit(node)  # type: ignore[return-value]

    @staticmethod
    def _is_self_module_call(node: ast.Call) -> bool:
        """Check if this is a self.module(x) call."""
        return (isinstance(node.func.value, ast.Name) and  # type: ignore[attr-defined]
                node.func.value.id == "self")  # type: ignore[attr-defined]

    def _is_functional_call(self, node: ast.Call) -> bool:
        """Check if this is a F.function(x) or torch.function(x) call."""
        module_path: list[str] = []
        inner_node = node.func.value  # type: ignore[attr-defined]
        while isinstance(inner_node, ast.Attribute):
            module_path.append(inner_node.attr)
            inner_node = inner_node.value
        if isinstance(inner_node, ast.Name):
            if inner_node.id in self.current_imports:
                module_path.append(self.current_imports[inner_node.id])
            else:
                if inner_node.id == "torch":
                    module_path.append("torch")
                else:
                    return False
        module_path.reverse()
        return ".".join(module_path) in ["torch.nn.functional", "torch"]

    @staticmethod
    def _is_pytorch_module_init(node: ast.Call) -> bool:
        """Check if this is a PyTorch module initialization."""
        if isinstance(node.func, ast.Attribute):
            module_name = node.func.attr
            return is_supported_module(module_name)
        return False

    @staticmethod
    def _transform_module_call(node: ast.Call, current_class_params: dict[str, dict[str, str | ast.Call]]) -> ast.Call | ast.Tuple:
        """Transform self.module(x) -> direct call or jax.vmap(self.module)(x) based on module type."""
        module_name = node.func.attr  # type: ignore[attr-defined]

        # Check if this is an RNN call that needs special handling
        if PyTorchToEquinoxTransformer._is_rnn_module_call(module_name, current_class_params):
            return PyTorchToEquinoxTransformer._transform_rnn_call(node, module_name, current_class_params)

        # Check if this module handles batches natively (Conv2d, Linear, etc.)
        if PyTorchToEquinoxTransformer._module_handles_batches_natively(module_name, current_class_params):
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

    @staticmethod
    def _transform_functional_call(node: ast.Call) -> ast.Call:
        """Transform F.function(x) -> jax.nn.function(x) using modular converters."""
        module_name = node.func.value.id  # type: ignore[attr-defined]
        func_name = node.func.attr  # type: ignore[attr-defined]

        # Use the modular functional converter
        from equitorch.converters import (
            transform_functional_call,  # import inside this function so that mock test works
        )
        return transform_functional_call(node, module_name, func_name)

    @staticmethod
    def _transform_method_call(node: ast.Call) -> ast.Call:
        """Transform x.method() calls using modular converters."""
        method_name = get_method_name(node)
        if method_name:
            from equitorch.converters import (
                transform_method_call,  # import inside this function so that mock test works
            )
            return transform_method_call(node, method_name)
        return node

    @staticmethod
    def _is_rnn_module_call(module_name: str, current_class_params: dict[str, dict[str, str | ast.Call]]) -> bool:
        """Check if this is a call to an RNN-like module that needs scan."""
        # Extended RNN type mapping
        RNN_MODULE_TYPES = {"RNN", "LSTM", "GRU", "RNNCell", "LSTMCell", "GRUCell"}

        # Check if the module is an RNN type based on our parameter tracking
        return module_name in current_class_params and current_class_params[module_name]["type"] in RNN_MODULE_TYPES


    @staticmethod
    def _module_handles_batches_natively(module_name: str, current_class_params: dict[str, dict[str, str | ast.Call]]) -> bool:
        """Check if a module handles batches natively and doesn't need vmap."""
        # Get the module type for this parameter
        if module_name not in current_class_params:
            return False
        module_type = current_class_params[module_name]["type"]

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
        from .converters.module_converters import get_supported_modules
        supported_modules = get_supported_modules()
        if module_type not in supported_modules:
            # This is a custom class, treat it as batch-native (no vmap needed)
            return True

        return False

    @staticmethod
    def _get_rnn_type(module_name: str, current_class_params: dict[str, dict[str, str | ast.Call]]) -> str:
        """Get the RNN type for a given module name."""
        return current_class_params[module_name]["type"] if module_name in current_class_params else "RNN"

    @staticmethod
    def _create_rnn_scan_function(rnn_type: str, module_name: str) -> ast.Lambda:
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

    @staticmethod
    def _transform_rnn_call(node: ast.Call, module_name: str, current_class_params: dict[str, dict[str, str | ast.Call]]) -> ast.Tuple:
        """Transform RNN calls to use jax.lax.scan for sequential processing."""
        # For RNN calls like self.rnn(x, hidden), we need to use jax.lax.scan
        # to process the sequence step by step

        # Get RNN type for this module
        rnn_type = PyTorchToEquinoxTransformer._get_rnn_type(module_name, current_class_params)

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
        scan_lambda = PyTorchToEquinoxTransformer._create_rnn_scan_function(rnn_type, module_name)

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

    @staticmethod
    def _store_type_annotation(original_node: ast.Call, type_annotation: ast.AST, current_class_params: dict[str, dict[str, str | ast.Call]]):
        """Store type annotation for a parameter based on its original call node."""
        # Find the parameter that corresponds to this call node
        for param in current_class_params.values():
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

    @staticmethod
    def _assignment_needs_key_splitting(node: ast.Assign) -> bool:
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

    def _is_custom_class_init(self, node: ast.Call, current_class_params: dict[str, dict[str, str | ast.Call]]) -> bool:
        """Check if this is a custom class instantiation that needs a key."""
        if isinstance(node.func, ast.Name):
            # Check if this class name is in our tracked parameters
            class_name = node.func.id
            for param in current_class_params.values():
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
        # Only add key to actual Call nodes, not function references
        if not isinstance(call_node, ast.Call):
            return
            
        # Check if key parameter already exists
        has_key = any(kw.arg == "key" for kw in call_node.keywords if kw.arg is not None)

        if not has_key:
            # Add the subkey to the call - the key splitting will be handled in visit_Assign
            key_keyword = ast.keyword(
                arg="key",
                value=ast.Name(id="subkey", ctx=ast.Load())
            )
            call_node.keywords.append(key_keyword)

    def _transform_module_init(self, node: ast.Call, current_class_params: dict[str, dict[str, str | ast.Call]]) -> ast.Call:
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
                self._store_type_annotation(node, type_annotation, current_class_params)
                # Don't add key parameter for Parameter, Dropout, Sequential, or LayerNorm
                if module_type not in ["Parameter", "Dropout", "Sequential", "LayerNorm"]:
                    self._add_key_to_call(transformed_node)
                return transformed_node
            else:
                # Old interface - add key parameter (except for Parameter and Dropout)
                if module_type not in ["Parameter", "Dropout"]:
                    self._add_key_to_call(result)
                return result
        return node

    def _transform_keyword_arguments(self, node: ast.Call) -> ast.Call:
        """Transform keyword arguments that contain PyTorch constructs.
        
        This handles cases like dtype=torch.float -> dtype=jnp.float32
        """
        for keyword in node.keywords:
            if keyword.value and isinstance(keyword.value, ast.Attribute):
                # Check for torch.dtype patterns
                if (isinstance(keyword.value.value, ast.Name) and
                    keyword.value.value.id == "torch"):
                    attr_name = keyword.value.attr
                    # Use functional mappings to convert torch.float -> jnp.float32
                    from .converters.functional_converters import FUNCTIONAL_MAPPINGS
                    mapping_key = ("torch", attr_name)
                    if mapping_key in FUNCTIONAL_MAPPINGS:
                        new_module, new_attr = FUNCTIONAL_MAPPINGS[mapping_key]
                        # Create new attribute reference
                        if "." in new_module:
                            module_parts = new_module.split(".")
                            keyword.value = ast.Attribute(
                                value=ast.Attribute(
                                    value=ast.Name(id=module_parts[0], ctx=ast.Load()),
                                    attr=module_parts[1],
                                    ctx=ast.Load(),
                                ),
                                attr=new_attr,
                                ctx=ast.Load(),
                            )
                        else:
                            keyword.value = ast.Attribute(
                                value=ast.Name(id=new_module, ctx=ast.Load()),
                                attr=new_attr,
                                ctx=ast.Load(),
                            )
        return node

    def _get_required_class_definitions(self) -> list[ast.stmt]:
        """Generate AST nodes for required class definitions using the unified system."""
        from .converters.module_converters import (
            get_required_custom_implementations,
            auto_transpile_module,
            get_supported_modules,
        )

        class_definitions = []

        # Get custom implementations for PyTorch modules that need them
        custom_implementations = get_required_custom_implementations(self.required_class_definitions)

        for custom_name, definition in custom_implementations.items():
            class_code = definition.strip()
            class_ast = ast.parse(class_code)
            class_definitions.extend(class_ast.body)

        # Handle auto-transpiled modules (those not in our configured modules)
        supported_modules = get_supported_modules()
        for pytorch_module in self.required_class_definitions:
            if pytorch_module not in supported_modules:
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
