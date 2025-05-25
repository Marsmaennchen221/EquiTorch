"""
AST builders for constructing Equinox code syntax trees.

This module provides builder classes that construct AST nodes for different
Equinox constructs, replacing direct string generation with proper AST construction.
"""

import ast
from typing import List, Dict, Any, Optional, Union


class EquinoxASTBuilder:
    """Base class for building Equinox AST nodes."""
    
    @staticmethod
    def create_import(module: str, alias: Optional[str] = None) -> ast.Import:
        """Create an import statement AST node."""
        if alias:
            return ast.Import(names=[ast.alias(name=module, asname=alias)])
        else:
            return ast.Import(names=[ast.alias(name=module, asname=None)])
    
    @staticmethod
    def create_name(name: str) -> ast.Name:
        """Create a Name AST node."""
        return ast.Name(id=name, ctx=ast.Load())
    
    @staticmethod
    def create_attribute(value: Union[str, ast.AST], attr: str) -> ast.Attribute:
        """Create an Attribute AST node."""
        if isinstance(value, str):
            value = EquinoxASTBuilder.create_name(value)
        return ast.Attribute(value=value, attr=attr, ctx=ast.Load())
    
    @staticmethod
    def create_constant(value: Any) -> ast.Constant:
        """Create a Constant AST node."""
        return ast.Constant(value=value)
    
    @staticmethod
    def create_call(func: Union[str, ast.AST], args: List[ast.AST] = None, 
                   keywords: List[ast.keyword] = None) -> ast.Call:
        """Create a Call AST node."""
        if isinstance(func, str):
            # Handle dotted names like "jax.nn.relu"
            if '.' in func:
                parts = func.split('.')
                func_ast = EquinoxASTBuilder.create_name(parts[0])
                for part in parts[1:]:
                    func_ast = EquinoxASTBuilder.create_attribute(func_ast, part)
            else:
                func_ast = EquinoxASTBuilder.create_name(func)
        else:
            func_ast = func
        
        return ast.Call(
            func=func_ast,
            args=args or [],
            keywords=keywords or []
        )
    
    @staticmethod
    def create_keyword(arg: str, value: ast.AST) -> ast.keyword:
        """Create a keyword argument AST node."""
        return ast.keyword(arg=arg, value=value)
    
    @staticmethod
    def create_assign(target: str, value: ast.AST) -> ast.Assign:
        """Create an assignment AST node."""
        return ast.Assign(
            targets=[ast.Name(id=target, ctx=ast.Store())],
            value=value
        )
    
    @staticmethod
    def create_attribute_assign(obj: str, attr: str, value: ast.AST) -> ast.Assign:
        """Create an attribute assignment AST node (e.g., self.conv1 = ...)."""
        return ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id=obj, ctx=ast.Load()),
                attr=attr,
                ctx=ast.Store()
            )],
            value=value
        )


class EquinoxClassBuilder:
    """Builder for Equinox module class AST nodes."""
    
    def __init__(self, class_name: str):
        self.class_name = class_name
        self.type_annotations = []
        self.init_args = []
        self.init_body = []
        self.call_body = []
    
    def add_type_annotation(self, name: str, type_annotation: str) -> 'EquinoxClassBuilder':
        """Add a type annotation for a class attribute."""
        # Create type annotation AST node
        if '.' in type_annotation:
            parts = type_annotation.split('.')
            annotation = EquinoxASTBuilder.create_name(parts[0])
            for part in parts[1:]:
                annotation = EquinoxASTBuilder.create_attribute(annotation, part)
        else:
            annotation = EquinoxASTBuilder.create_name(type_annotation)
        
        ann_assign = ast.AnnAssign(
            target=ast.Name(id=name, ctx=ast.Store()),
            annotation=annotation,
            value=None,
            simple=1
        )
        self.type_annotations.append(ann_assign)
        return self
    
    def add_init_arg(self, arg_name: str) -> 'EquinoxClassBuilder':
        """Add an argument to the __init__ method."""
        self.init_args.append(arg_name)
        return self
    
    def add_init_statement(self, stmt: ast.AST) -> 'EquinoxClassBuilder':
        """Add a statement to the __init__ method body."""
        self.init_body.append(stmt)
        return self
    
    def add_call_statement(self, stmt: ast.AST) -> 'EquinoxClassBuilder':
        """Add a statement to the __call__ method body."""
        self.call_body.append(stmt)
        return self
    
    def build(self) -> ast.ClassDef:
        """Build the complete class AST node."""
        # Create base class (eqx.Module)
        base = EquinoxASTBuilder.create_attribute("eqx", "Module")
        
        # Build __init__ method
        init_args = [ast.arg(arg="self", annotation=None)]
        for arg_name in self.init_args:
            init_args.append(ast.arg(arg=arg_name, annotation=None))
        
        # Add default arguments for key and **kwargs
        init_args.append(ast.arg(arg="key", annotation=None))
        
        init_defaults = [
            EquinoxASTBuilder.create_constant(None)  # key=None
        ]
        
        init_method = ast.FunctionDef(
            name="__init__",
            args=ast.arguments(
                posonlyargs=[],
                args=init_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=ast.arg(arg="kwargs", annotation=None),
                defaults=init_defaults
            ),
            body=self.init_body,
            decorator_list=[],
            returns=None
        )
        
        # Build __call__ method
        call_args = [
            ast.arg(arg="self", annotation=None),
            ast.arg(arg="x", annotation=None)
        ]
        
        call_method = ast.FunctionDef(
            name="__call__",
            args=ast.arguments(
                posonlyargs=[],
                args=call_args,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=self.call_body,
            decorator_list=[],
            returns=None
        )
        
        # Combine all class body elements
        class_body = self.type_annotations + [init_method, call_method]
        
        return ast.ClassDef(
            name=self.class_name,
            bases=[base],
            keywords=[],
            decorator_list=[],
            body=class_body
        )


class EquinoxCallBuilder:
    """Builder for JAX/Equinox function call AST nodes."""
    
    @staticmethod
    def create_jax_vmap_call(module_attr: str, input_var: str) -> ast.Call:
        """Create a jax.vmap(self.module)(input) call."""
        # Create jax.vmap
        vmap_func = EquinoxASTBuilder.create_attribute("jax", "vmap")
        
        # Create self.module
        module_ref = EquinoxASTBuilder.create_attribute("self", module_attr)
        
        # Create jax.vmap(self.module)
        vmap_call = EquinoxASTBuilder.create_call(vmap_func, [module_ref])
        
        # Create the final call with input
        input_ref = EquinoxASTBuilder.create_name(input_var)
        return EquinoxASTBuilder.create_call(vmap_call, [input_ref])
    
    @staticmethod
    def create_jax_nn_call(func_name: str, input_var: str, **kwargs) -> ast.Call:
        """Create a jax.nn.function(input, **kwargs) call."""
        func = EquinoxASTBuilder.create_attribute(
            EquinoxASTBuilder.create_attribute("jax", "nn"), 
            func_name
        )
        
        args = [EquinoxASTBuilder.create_name(input_var)]
        keywords = []
        
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str, bool)):
                kw_value = EquinoxASTBuilder.create_constant(value)
            elif isinstance(value, (list, tuple)):
                # Handle tuples/lists for things like window_shape
                elements = [EquinoxASTBuilder.create_constant(v) for v in value]
                if isinstance(value, tuple):
                    kw_value = ast.Tuple(elts=elements, ctx=ast.Load())
                else:
                    kw_value = ast.List(elts=elements, ctx=ast.Load())
            else:
                kw_value = EquinoxASTBuilder.create_name(str(value))
            
            keywords.append(EquinoxASTBuilder.create_keyword(key, kw_value))
        
        return EquinoxASTBuilder.create_call(func, args, keywords)
    
    @staticmethod
    def create_jnp_reshape_call(input_var: str, shape_args: List[Any]) -> ast.Call:
        """Create a jnp.reshape(input, shape) call."""
        func = EquinoxASTBuilder.create_attribute("jnp", "reshape")
        
        args = [EquinoxASTBuilder.create_name(input_var)]
        
        # Build shape tuple
        shape_elements = []
        for arg in shape_args:
            if isinstance(arg, (int, float)):
                shape_elements.append(EquinoxASTBuilder.create_constant(arg))
            elif isinstance(arg, str):
                if arg == "-1":
                    shape_elements.append(EquinoxASTBuilder.create_constant(-1))
                elif ".shape[" in arg:
                    # Handle expressions like "x.shape[0]"
                    parts = arg.split(".shape[")
                    var_name = parts[0]
                    index = int(parts[1].rstrip("]"))
                    
                    shape_attr = EquinoxASTBuilder.create_attribute(var_name, "shape")
                    subscript = ast.Subscript(
                        value=shape_attr,
                        slice=EquinoxASTBuilder.create_constant(index),
                        ctx=ast.Load()
                    )
                    shape_elements.append(subscript)
                else:
                    shape_elements.append(EquinoxASTBuilder.create_name(arg))
            else:
                shape_elements.append(EquinoxASTBuilder.create_constant(arg))
        
        shape_tuple = ast.Tuple(elts=shape_elements, ctx=ast.Load())
        args.append(shape_tuple)
        
        return EquinoxASTBuilder.create_call(func, args)


class EquinoxModuleBuilder:
    """Builder for complete Equinox module AST."""
    
    def __init__(self):
        self.imports = []
        self.classes = []
    
    def add_import(self, module: str, alias: Optional[str] = None) -> 'EquinoxModuleBuilder':
        """Add an import statement."""
        self.imports.append(EquinoxASTBuilder.create_import(module, alias))
        return self
    
    def add_class(self, class_def: ast.ClassDef) -> 'EquinoxModuleBuilder':
        """Add a class definition."""
        self.classes.append(class_def)
        return self
    
    def build(self) -> ast.Module:
        """Build the complete module AST."""
        body = self.imports + self.classes
        return ast.Module(body=body, type_ignores=[])
