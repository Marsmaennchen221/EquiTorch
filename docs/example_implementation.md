# Supporting additional Pytorch functionalities
This document leads the user through an exemplary implementaiton to support a new pytorch object that was formerly unsupported by the transpiler.

# Example
We start with the observation of an unsupported Pytorch functionality. For this simple case we look at the following implementation of a mulit-layer-perceptron:

```Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        
        # Hidden layer parameters
        self.w1 = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden_size))
        
        # Output layer parameters
        self.w2 = nn.Parameter(torch.randn(output_size, hidden_size) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        # Hidden layer with ReLU activation
        hidden = F.linear(x, self.w1, self.b1)
        hidden = F.relu(hidden)
        
        # Output layer
        output = F.linear(hidden, self.w2, self.b2)
        
        return output
```

It utilizes Pytorch's functional implementation of the linear layer: `F.linear(...)`, which is currently not supported. To add support for it, we need to process the following steps:

1. Add a custom transformation function for the mapping, since `F.linear()` has no preimplemented equivalent in equinox.
```Python
CUSTOM_FUNCTIONAL_TRANSFORMERS: dict[tuple[str, str], callable] = {
    # ...
    ("F", "linear"): lambda node: transform_linear(node),
    # ...
}
```
2. Implement the custom transformation function, that constructs the AST of the target equinox implementation.

```Python
def transform_linear(node: ast.Call) -> ast.Call:
    """ jnp.dot(input, weight.T) + bias """
    
    input_tensor = node.args[0]
    weight = node.args[1]
    bias = node.args[2]
    # Create weight.T (transpose)
    weight_transposed = ast.Attribute(
        value=weight,
        attr="T",
        ctx=ast.Load(),
    )
    # Create jnp.dot(input, weight.T)
    dot_result = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="jnp", ctx=ast.Load()),
            attr="dot",
            ctx=ast.Load(),
        ),
        args=[input_tensor, weight_transposed],
        keywords=[],
    )
    # Add bias: jnp.dot(input, weight.T) + bias
    return ast.BinOp(
        left=dot_result,
        op=ast.Add(),
        right=bias,
    )
    else:
        return dot_result
```

In this version we directly construct the AST of our target implementation by manipulating the existing AST and adding new nodes to implement the operations we need. 
Another way to inject the neccessary AST structure is to create it by unparsing the target code string into the AST of the wished construct.

```Python
def transform_linear(node: ast.Call) -> ast.Call:
    """ jnp.dot(input, weight.T) + bias (if bias is not None) """

    # Extract arguments
    input_tensor = node.args[0]
    weight = node.args[1]
    bias = node.args[2] 

    # Convert AST nodes to code strings
    input_str = ast.unparse(input_tensor)
    weight_str = ast.unparse(weight)
    
    # Build target code string
    bias_str = ast.unparse(bias)
    target_code = f"jnp.dot({input_str}, {weight_str}.T) + {bias_str}"
    
    # Parse back to AST and return the expression
    parsed = ast.parse(target_code, mode='eval')
    return parsed.body
```

Using AST nodes as "interface" between different parts of the transpiler allows us to be very flexible in the way a certain pytorch component translation is provided. It comes with the drawback of potential code repition e.g. for extracting arguments, and maybe handling nested structures within those arguments, but it does not put a considerable burdon on the person implementing a new translation and trying to solve this generically for every potential scenario would create an unnecessary complex extraction and reinjection system that is probably more difficult to understand for the implementing person than it actually helps. Therefore I beleive that this is fine.
