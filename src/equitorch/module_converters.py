import ast
from typing import Dict, Any, Optional, List
from .ast_builders import EquinoxASTBuilder

class ModuleConverter:
    """Base class for module converters that transform PyTorch modules to Equinox."""
    
    def extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """
        Extract arguments from a module initialization call.
        
        Args:
            call_node: AST call node representing the module initialization
            
        Returns:
            Dictionary of argument names and values
        """
        raise NotImplementedError("Subclasses must implement extract_args")
    
    def generate_equinox_code(self, param_name: str, args: Dict[str, Any], key_var: str) -> str:
        """
        Generate Equinox code for initializing this module.
        
        Args:
            param_name: Name of the parameter (e.g., 'self.conv1')
            args: Dictionary of arguments extracted from extract_args
            key_var: Variable name for the PRNG key
            
        Returns:
            String containing Equinox initialization code
        """
        raise NotImplementedError("Subclasses must implement generate_equinox_code")
    
    def generate_equinox_ast(self, param_name: str, args: Dict[str, Any], key_var: str) -> ast.Assign:
        """
        Generate Equinox AST node for initializing this module.
        
        Args:
            param_name: Name of the parameter (e.g., 'self.conv1')
            args: Dictionary of arguments extracted from extract_args
            key_var: Variable name for the PRNG key
            
        Returns:
            AST assignment node for Equinox initialization
        """
        raise NotImplementedError("Subclasses must implement generate_equinox_ast")
    
    def get_equinox_type(self) -> str:
        """
        Return the Equinox type annotation for this module.
        
        Returns:
            String representing the Equinox type (e.g., 'eqx.nn.Linear')
        """
        raise NotImplementedError("Subclasses must implement get_equinox_type")
    
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


class LinearConverter(ModuleConverter):
    """Converter for nn.Linear to eqx.nn.Linear."""
    
    def extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from nn.Linear initialization."""
        args = {}
        
        # Extract positional arguments
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
        
        return args
    
    def generate_equinox_code(self, param_name: str, args: Dict[str, Any], key_var: str) -> str:
        """Generate Equinox code for initializing a Linear layer."""
        in_features = args.get('in_features', '...')
        out_features = args.get('out_features', '...')
        bias = args.get('bias', 'True')
        
        return f"{param_name} = eqx.nn.Linear({in_features}, {out_features}, use_bias={bias}, key={key_var})"
    
    def generate_equinox_ast(self, param_name: str, args: Dict[str, Any], key_var: str) -> ast.Assign:
        """Generate Equinox AST node for initializing a Linear layer."""
        in_features = args.get('in_features', '...')
        out_features = args.get('out_features', '...')
        bias = args.get('bias', True)
        
        # Create arguments for eqx.nn.Linear call
        call_args = []
        
        # Add positional arguments
        if isinstance(in_features, str) and in_features != '...':
            call_args.append(EquinoxASTBuilder.create_name(in_features))
        elif isinstance(in_features, (int, float)):
            call_args.append(EquinoxASTBuilder.create_constant(in_features))
        
        if isinstance(out_features, str) and out_features != '...':
            call_args.append(EquinoxASTBuilder.create_name(out_features))
        elif isinstance(out_features, (int, float)):
            call_args.append(EquinoxASTBuilder.create_constant(out_features))
        
        # Add keyword arguments
        keywords = []
        if isinstance(bias, bool):
            keywords.append(EquinoxASTBuilder.create_keyword('use_bias', EquinoxASTBuilder.create_constant(bias)))
        elif isinstance(bias, str) and bias.lower() in ['true', 'false']:
            keywords.append(EquinoxASTBuilder.create_keyword('use_bias', EquinoxASTBuilder.create_constant(bias.lower() == 'true')))
        
        keywords.append(EquinoxASTBuilder.create_keyword('key', EquinoxASTBuilder.create_name(key_var)))
        
        # Create the eqx.nn.Linear call
        linear_call = EquinoxASTBuilder.create_call('eqx.nn.Linear', call_args, keywords)
        
        # Create the assignment
        if param_name.startswith('self.'):
            attr_name = param_name[5:]  # Remove 'self.' prefix
            return EquinoxASTBuilder.create_attribute_assign('self', attr_name, linear_call)
        else:
            return EquinoxASTBuilder.create_assign(param_name, linear_call)
    
    def get_equinox_type(self) -> str:
        """Return the Equinox type for Linear."""
        return "eqx.nn.Linear"


class Conv2dConverter(ModuleConverter):
    """Converter for nn.Conv2d to eqx.nn.Conv2d."""
    
    def extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from nn.Conv2d initialization."""
        args = {}
        
        # Extract positional arguments
        for i, arg in enumerate(call_node.args):
            if i == 0:
                args['in_channels'] = self._extract_value(arg)
            elif i == 1:
                args['out_channels'] = self._extract_value(arg)
            elif i == 2:
                args['kernel_size'] = self._extract_value(arg)
        
        # Extract keyword arguments
        for kw in call_node.keywords:
            if kw.arg == 'in_channels':
                args['in_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'out_channels':
                args['out_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'kernel_size':
                args['kernel_size'] = self._extract_value(kw.value)
            elif kw.arg == 'stride':
                args['stride'] = self._extract_value(kw.value)
            elif kw.arg == 'padding':
                args['padding'] = self._extract_value(kw.value)
            elif kw.arg == 'dilation':
                args['dilation'] = self._extract_value(kw.value)
            elif kw.arg == 'groups':
                args['groups'] = self._extract_value(kw.value)
            elif kw.arg == 'bias':
                args['bias'] = self._extract_value(kw.value)
            elif kw.arg == 'padding_mode':
                args['padding_mode'] = self._extract_value(kw.value)
        
        return args
    
    def generate_equinox_code(self, param_name: str, args: Dict[str, Any], key_var: str) -> str:
        """Generate Equinox code for initializing a Conv2d layer."""
        in_channels = args.get('in_channels', '...')
        out_channels = args.get('out_channels', '...')
        kernel_size = args.get('kernel_size', '...')
        stride = args.get('stride', '1')
        padding = args.get('padding', '0')
        dilation = args.get('dilation', '1')
        groups = args.get('groups', '1')
        bias = args.get('bias', 'True')
        
        return (f"{param_name} = eqx.nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, "
                f"stride={stride}, padding={padding}, dilation={dilation}, "
                f"groups={groups}, use_bias={bias}, key={key_var})")
    
    def generate_equinox_ast(self, param_name: str, args: Dict[str, Any], key_var: str) -> ast.Assign:
        """Generate Equinox AST node for initializing a Conv2d layer."""
        in_channels = args.get('in_channels', '...')
        out_channels = args.get('out_channels', '...')
        kernel_size = args.get('kernel_size', '...')
        stride = args.get('stride', 1)
        padding = args.get('padding', 0)
        dilation = args.get('dilation', 1)
        groups = args.get('groups', 1)
        bias = args.get('bias', True)
        
        # Create arguments for eqx.nn.Conv2d call
        call_args = []
        
        # Add positional arguments
        if isinstance(in_channels, str) and in_channels != '...':
            call_args.append(EquinoxASTBuilder.create_name(in_channels))
        elif isinstance(in_channels, (int, float)):
            call_args.append(EquinoxASTBuilder.create_constant(in_channels))
        
        if isinstance(out_channels, str) and out_channels != '...':
            call_args.append(EquinoxASTBuilder.create_name(out_channels))
        elif isinstance(out_channels, (int, float)):
            call_args.append(EquinoxASTBuilder.create_constant(out_channels))
        
        # Add keyword arguments
        keywords = []
        
        # kernel_size
        if isinstance(kernel_size, str) and kernel_size != '...':
            keywords.append(EquinoxASTBuilder.create_keyword('kernel_size', EquinoxASTBuilder.create_name(kernel_size)))
        elif isinstance(kernel_size, (int, float)):
            keywords.append(EquinoxASTBuilder.create_keyword('kernel_size', EquinoxASTBuilder.create_constant(kernel_size)))
        
        # stride
        if isinstance(stride, (int, float)):
            keywords.append(EquinoxASTBuilder.create_keyword('stride', EquinoxASTBuilder.create_constant(stride)))
        elif isinstance(stride, str):
            keywords.append(EquinoxASTBuilder.create_keyword('stride', EquinoxASTBuilder.create_name(stride)))
        
        # padding
        if isinstance(padding, (int, float)):
            keywords.append(EquinoxASTBuilder.create_keyword('padding', EquinoxASTBuilder.create_constant(padding)))
        elif isinstance(padding, str):
            keywords.append(EquinoxASTBuilder.create_keyword('padding', EquinoxASTBuilder.create_name(padding)))
        
        # dilation
        if isinstance(dilation, (int, float)):
            keywords.append(EquinoxASTBuilder.create_keyword('dilation', EquinoxASTBuilder.create_constant(dilation)))
        elif isinstance(dilation, str):
            keywords.append(EquinoxASTBuilder.create_keyword('dilation', EquinoxASTBuilder.create_name(dilation)))
        
        # groups
        if isinstance(groups, (int, float)):
            keywords.append(EquinoxASTBuilder.create_keyword('groups', EquinoxASTBuilder.create_constant(groups)))
        elif isinstance(groups, str):
            keywords.append(EquinoxASTBuilder.create_keyword('groups', EquinoxASTBuilder.create_name(groups)))
        
        # use_bias
        if isinstance(bias, bool):
            keywords.append(EquinoxASTBuilder.create_keyword('use_bias', EquinoxASTBuilder.create_constant(bias)))
        elif isinstance(bias, str) and bias.lower() in ['true', 'false']:
            keywords.append(EquinoxASTBuilder.create_keyword('use_bias', EquinoxASTBuilder.create_constant(bias.lower() == 'true')))
        
        keywords.append(EquinoxASTBuilder.create_keyword('key', EquinoxASTBuilder.create_name(key_var)))
        
        # Create the eqx.nn.Conv2d call
        conv_call = EquinoxASTBuilder.create_call('eqx.nn.Conv2d', call_args, keywords)
        
        # Create the assignment
        if param_name.startswith('self.'):
            attr_name = param_name[5:]  # Remove 'self.' prefix
            return EquinoxASTBuilder.create_attribute_assign('self', attr_name, conv_call)
        else:
            return EquinoxASTBuilder.create_assign(param_name, conv_call)
    
    def get_equinox_type(self) -> str:
        """Return the Equinox type for Conv2d."""
        return "eqx.nn.Conv2d"


class Conv1dConverter(ModuleConverter):
    """Converter for nn.Conv1d to eqx.nn.Conv1d."""
    
    def extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from nn.Conv1d initialization."""
        args = {}
        
        # Extract positional arguments
        for i, arg in enumerate(call_node.args):
            if i == 0:
                args['in_channels'] = self._extract_value(arg)
            elif i == 1:
                args['out_channels'] = self._extract_value(arg)
            elif i == 2:
                args['kernel_size'] = self._extract_value(arg)
        
        # Extract keyword arguments
        for kw in call_node.keywords:
            if kw.arg == 'in_channels':
                args['in_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'out_channels':
                args['out_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'kernel_size':
                args['kernel_size'] = self._extract_value(kw.value)
            elif kw.arg == 'stride':
                args['stride'] = self._extract_value(kw.value)
            elif kw.arg == 'padding':
                args['padding'] = self._extract_value(kw.value)
            elif kw.arg == 'dilation':
                args['dilation'] = self._extract_value(kw.value)
            elif kw.arg == 'groups':
                args['groups'] = self._extract_value(kw.value)
            elif kw.arg == 'bias':
                args['bias'] = self._extract_value(kw.value)
            elif kw.arg == 'padding_mode':
                args['padding_mode'] = self._extract_value(kw.value)
        
        return args
    
    def generate_equinox_code(self, param_name: str, args: Dict[str, Any], key_var: str) -> str:
        """Generate Equinox code for initializing a Conv1d layer."""
        in_channels = args.get('in_channels', '...')
        out_channels = args.get('out_channels', '...')
        kernel_size = args.get('kernel_size', '...')
        stride = args.get('stride', '1')
        padding = args.get('padding', '0')
        dilation = args.get('dilation', '1')
        groups = args.get('groups', '1')
        bias = args.get('bias', 'True')
        
        return (f"{param_name} = eqx.nn.Conv1d({in_channels}, {out_channels}, kernel_size={kernel_size}, "
                f"stride={stride}, padding={padding}, dilation={dilation}, "
                f"groups={groups}, use_bias={bias}, key={key_var})")
    
    def get_equinox_type(self) -> str:
        """Return the Equinox type for Conv1d."""
        return "eqx.nn.Conv1d"


class Conv3dConverter(ModuleConverter):
    """Converter for nn.Conv3d to eqx.nn.Conv3d."""
    
    def extract_args(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from nn.Conv3d initialization."""
        args = {}
        
        # Extract positional arguments
        for i, arg in enumerate(call_node.args):
            if i == 0:
                args['in_channels'] = self._extract_value(arg)
            elif i == 1:
                args['out_channels'] = self._extract_value(arg)
            elif i == 2:
                args['kernel_size'] = self._extract_value(arg)
        
        # Extract keyword arguments
        for kw in call_node.keywords:
            if kw.arg == 'in_channels':
                args['in_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'out_channels':
                args['out_channels'] = self._extract_value(kw.value)
            elif kw.arg == 'kernel_size':
                args['kernel_size'] = self._extract_value(kw.value)
            elif kw.arg == 'stride':
                args['stride'] = self._extract_value(kw.value)
            elif kw.arg == 'padding':
                args['padding'] = self._extract_value(kw.value)
            elif kw.arg == 'dilation':
                args['dilation'] = self._extract_value(kw.value)
            elif kw.arg == 'groups':
                args['groups'] = self._extract_value(kw.value)
            elif kw.arg == 'bias':
                args['bias'] = self._extract_value(kw.value)
            elif kw.arg == 'padding_mode':
                args['padding_mode'] = self._extract_value(kw.value)
        
        return args
    
    def generate_equinox_code(self, param_name: str, args: Dict[str, Any], key_var: str) -> str:
        """Generate Equinox code for initializing a Conv3d layer."""
        in_channels = args.get('in_channels', '...')
        out_channels = args.get('out_channels', '...')
        kernel_size = args.get('kernel_size', '...')
        stride = args.get('stride', '1')
        padding = args.get('padding', '0')
        dilation = args.get('dilation', '1')
        groups = args.get('groups', '1')
        bias = args.get('bias', 'True')
        
        return (f"{param_name} = eqx.nn.Conv3d({in_channels}, {out_channels}, kernel_size={kernel_size}, "
                f"stride={stride}, padding={padding}, dilation={dilation}, "
                f"groups={groups}, use_bias={bias}, key={key_var})")
    
    def get_equinox_type(self) -> str:
        """Return the Equinox type for Conv3d."""
        return "eqx.nn.Conv3d"
