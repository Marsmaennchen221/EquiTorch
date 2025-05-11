import ast
from typing import Dict, Any, Optional, List

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
