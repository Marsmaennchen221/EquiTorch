"""
Complete transpiler using proper AST transformation patterns and modular converters.

Replaces the complex manual AST processing with clean, standard patterns
and uses separated converter modules for better organization.
"""
import ast
import sys
from typing import Optional, List

from .converters import (
    # Module converters
    is_supported_module,
    get_module_converter,
    
    # Functional converters
    FUNCTIONAL_MAPPINGS,
    transform_functional_call,
    
    # Method converters
    transform_method_call,
    is_tensor_method_call,
    get_method_name,
)


class PyTorchToEquinoxTransformer(ast.NodeTransformer):
    """
    Complete transpiler using proper AST transformation patterns and modular converters.

    This replaces the complex manual AST processing with clean, standard patterns
    and uses separated converter modules for better organization.
    """

    def __init__(self):
        """Initialize the transpiler."""
        self.current_class_params = []  # Track parameters for current class
        self.key_counter = 0
    
    def visit_Module(self, node: ast.Module) -> ast.Module:  # noqa: N802
        """Transform the entire module."""
        # Add imports at the top
        imports = [
            ast.Import(names=[ast.alias(name='jax', asname=None)]),
            ast.Import(names=[ast.alias(name='jax.numpy', asname='jnp')]),
            ast.Import(names=[ast.alias(name='equinox', asname='eqx')]),
        ]
        
        # Transform the rest of the module
        new_node: ast.Module = self.generic_visit(node)  # type: ignore[assignment]
        
        # Prepend imports to the body
        new_node.body = imports + new_node.body
        
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
            value=ast.Name(id='eqx', ctx=ast.Load()),
            attr='Module',
            ctx=ast.Load()
        )]
        
        # Add type annotations for parameters
        self._add_type_annotations(new_node)
        
        return new_node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:  # noqa: N802
        """Transform methods."""
        if node.name == '__init__':
            return self._transform_init_method(node)
        elif node.name == 'forward':
            return self._transform_forward_method(node)
        else:
            return node
    
    def visit_Call(self, node: ast.Call) -> ast.Call:  # noqa: N802
        """Using visitor pattern here now. nested calls are just handled recursively."""
        # First, recursively transform any nested calls
        node = self.generic_visit(node)  # type: ignore[assignment]
        
        # Now transform this specific call
        if isinstance(node.func, ast.Attribute):
            if self._is_self_module_call(node):
                return self._transform_module_call(node)
            elif self._is_functional_call(node):
                return self._transform_functional_call(node)
            elif is_tensor_method_call(node):
                return self._transform_method_call(node)
            elif self._is_pytorch_module_init(node):
                return self._transform_module_init(node)
        
        return node
    
    def _is_pytorch_module(self, node: ast.ClassDef) -> bool:
        """Check if class inherits from nn.Module."""
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == 'Module':
                return True
            elif isinstance(base, ast.Name) and base.id == 'Module':
                return True
        return False
    
    def _find_init_method(self, node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method."""
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
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
                    node.targets[0].value.id == 'self' and
                    isinstance(node.value, ast.Call)):
                    
                    param_name = node.targets[0].attr
                    if self.parent._is_pytorch_module_init(node.value):
                        module_type = node.value.func.attr
                        self.parent.current_class_params.append({
                            'name': param_name,
                            'type': module_type,
                            'call_node': node.value
                        })
        
        extractor = ParameterExtractor(self)
        extractor.visit(init_method)
    
    def _add_type_annotations(self, class_node: ast.ClassDef):
        """Add type annotations for parameters."""
        annotations = []
        for param in self.current_class_params:
            annotation = ast.AnnAssign(
                target=ast.Name(id=param['name'], ctx=ast.Store()),
                annotation=ast.Attribute(
                    value=ast.Attribute(
                        value=ast.Name(id='eqx', ctx=ast.Load()),
                        attr='nn',
                        ctx=ast.Load()
                    ),
                    attr=param['type'],
                    ctx=ast.Load()
                ),
                value=None,
                simple=1
            )
            annotations.append(annotation)
        
        # Insert annotations at the beginning of class body
        class_node.body = annotations + class_node.body
    
    def _transform_init_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform __init__ method."""
        # Add key parameter with default None
        if not any(arg.arg == 'key' for arg in node.args.args):
            node.args.args.append(ast.arg(arg='key', annotation=None))
            node.args.defaults.append(ast.Constant(value=None))
        
        # Add **kwargs if not present
        if not node.args.kwarg:
            node.args.kwarg = ast.arg(arg='kwargs', annotation=None)
        
        # Transform the body
        new_body: List[ast.stmt] = []
        
        # Add key initialization
        key_init = ast.If(
            test=ast.Compare(
                left=ast.Name(id='key', ctx=ast.Load()),
                ops=[ast.Is()],
                comparators=[ast.Constant(value=None)]
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id='key', ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id='jax', ctx=ast.Load()),
                                attr='random',
                                ctx=ast.Load()
                            ),
                            attr='PRNGKey',
                            ctx=ast.Load()
                        ),
                        args=[ast.Constant(value=0)],
                        keywords=[]
                    )
                )
            ],
            orelse=[]
        )
        new_body.append(key_init)
        
        # Transform existing body
        for stmt in node.body:
            new_body.append(self.visit(stmt))
        
        node.body = new_body
        return node
    
    def _transform_forward_method(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Transform forward method to __call__."""
        node.name = '__call__'
        
        # Transform the body - this automatically handles all nested expressions!
        return self.generic_visit(node)  # type: ignore[return-value]
    
    def _is_self_module_call(self, node: ast.Call) -> bool:
        """Check if this is a self.module(x) call."""
        return (isinstance(node.func.value, ast.Name) and  # type: ignore[attr-defined]
                node.func.value.id == 'self')  # type: ignore[attr-defined]
    
    def _is_functional_call(self, node: ast.Call) -> bool:
        """Check if this is a F.function(x) or torch.function(x) call."""
        return (isinstance(node.func.value, ast.Name) and  # type: ignore[attr-defined]
                node.func.value.id in ['F', 'torch'])  # type: ignore[attr-defined]
    
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
            value=ast.Name(id='jax', ctx=ast.Load()),
            attr='vmap',
            ctx=ast.Load()
        )
        
        module_ref = ast.Attribute(
            value=ast.Name(id='self', ctx=ast.Load()),
            attr=module_name,
            ctx=ast.Load()
        )
        
        vmap_call = ast.Call(
            func=vmap_func,
            args=[module_ref],
            keywords=[]
        )
        
        return ast.Call(
            func=vmap_call,
            args=node.args,
            keywords=node.keywords
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
            return converter(node, self.key_counter - 1)
        return node


def transpile(pytorch_code: str) -> str:
    """
    Complete transpiler using proper AST transformation patterns and modular converters.
    
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
