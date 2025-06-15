import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.nn as nn

class NestedCustomCNN(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm2d
    conv2: eqx.nn.Conv2d
    gn2: GroupNorm
    conv3: eqx.nn.Conv2d
    bn3: eqx.nn.BatchNorm2d
    adaptive_pool: AdaptiveAvgPool2d
    classifier: Sequential


    'A CNN module that uses multiple types of modules to test recursive transpilation.\n    \n    This module combines:\n    - Manual converters: Conv2d, BatchNorm2d, Linear, Dropout\n    - Auto-transpiled modules: GroupNorm, LayerNorm, AdaptiveAvgPool2d\n    '

    def __init__(self, input_channels, num_classes, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        key, subkey = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.bn1 = eqx.nn.BatchNorm2d(32, key=subkey)
        key, subkey = jax.random.split(key)
        self.conv2 = eqx.nn.Conv2d(32, 64, kernel_size=3, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.gn2 = GroupNorm(8, 64, key=subkey)
        key, subkey = jax.random.split(key)
        self.conv3 = eqx.nn.Conv2d(64, 128, kernel_size=3, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.bn3 = eqx.nn.BatchNorm2d(128, key=subkey)
        key, subkey = jax.random.split(key)
        self.adaptive_pool = AdaptiveAvgPool2d((4, 4), key=subkey)
        key, subkey = jax.random.split(key)
        self.classifier = Sequential(eqx.nn.Linear(128 * 4 * 4, 256, use_bias=True, key=subkey), LayerNorm(256, key=subkey), eqx.nn.Dropout(0.5, key=subkey), eqx.nn.Linear(256, num_classes, use_bias=True, key=subkey), key=subkey)

    def __call__(self, x):
        x = jax.vmap(self.conv1)(x)
        x = jax.vmap(self.bn1)(x)
        x = jax.nn.relu(x)
        x = jax.nn.max_pool(x, 2)
        x = jax.vmap(self.conv2)(x)
        x = jax.vmap(self.gn2)(x)
        x = jax.nn.relu(x)
        x = jax.nn.max_pool(x, 2)
        x = jax.vmap(self.conv3)(x)
        x = jax.vmap(self.bn3)(x)
        x = jax.nn.relu(x)
        x = jax.vmap(self.adaptive_pool)(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jax.vmap(self.classifier)(x)
        return x

class CustomResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    bn1: eqx.nn.BatchNorm2d
    conv2: eqx.nn.Conv2d
    gn2: GroupNorm
    skip_conv: eqx.nn.Conv2d
    'A custom residual block that uses auto-transpiled modules.'

    def __init__(self, in_channels, out_channels, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        key, subkey = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.bn1 = eqx.nn.BatchNorm2d(out_channels, key=subkey)
        key, subkey = jax.random.split(key)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.gn2 = GroupNorm(8, out_channels, key=subkey)
        self.skip_conv = None
        if in_channels != out_channels:
            key, subkey = jax.random.split(key)
            self.skip_conv = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, use_bias=True, key=subkey)

    def __call__(self, x):
        identity = x
        out = jax.vmap(self.conv1)(x)
        out = jax.vmap(self.bn1)(out)
        out = jax.nn.relu(out)
        out = jax.vmap(self.conv2)(out)
        out = jax.vmap(self.gn2)(out)
        if self.skip_conv is not None:
            identity = jax.vmap(self.skip_conv)(x)
        out += identity
        out = jax.nn.relu(out)
        return out

class DeepNestedCNN(eqx.Module):
    initial_conv: eqx.nn.Conv2d
    initial_bn: eqx.nn.BatchNorm2d
    block1: CustomResidualBlock
    block2: CustomResidualBlock
    block3: CustomResidualBlock
    global_pool: AdaptiveAvgPool2d
    classifier: Sequential
    
    'A deep CNN that nests the custom residual blocks to test deep recursion.'

    def __init__(self, input_channels, num_classes, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        key, subkey = jax.random.split(key)
        self.initial_conv = eqx.nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, use_bias=True, key=subkey)
        key, subkey = jax.random.split(key)
        self.initial_bn = eqx.nn.BatchNorm2d(64, key=subkey)
        key, subkey = jax.random.split(key)
        self.block1 = CustomResidualBlock(64, 64, key=subkey)
        key, subkey = jax.random.split(key)
        self.block2 = CustomResidualBlock(64, 128, key=subkey)
        key, subkey = jax.random.split(key)
        self.block3 = CustomResidualBlock(128, 256, key=subkey)
        key, subkey = jax.random.split(key)
        self.global_pool = AdaptiveAvgPool2d((1, 1), key=subkey)
        key, subkey = jax.random.split(key)
        self.classifier = Sequential(eqx.nn.Linear(256, 128, use_bias=True, key=subkey), LayerNorm(128, key=subkey), eqx.nn.Dropout(0.3, key=subkey), eqx.nn.Linear(128, num_classes, use_bias=True, key=subkey), key=subkey)

    def __call__(self, x):
        x = jax.vmap(self.initial_conv)(x)
        x = jax.vmap(self.initial_bn)(x)
        x = jax.nn.relu(x)
        x = jax.nn.max_pool(x, kernel_size=3, stride=2, padding=1)
        x = jax.vmap(self.block1)(x)
        x = jax.vmap(self.block2)(x)
        x = jax.vmap(self.block3)(x)
        x = jax.vmap(self.global_pool)(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        x = jax.vmap(self.classifier)(x)
        return x

class GroupNorm(eqx.Module):
    weight: Parameter
    bias: Parameter
    'Applies Group Normalization over a mini-batch of inputs.\n\n    This layer implements the operation as described in\n    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__\n\n    .. math::\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The input channels are separated into :attr:`num_groups` groups, each containing\n    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by\n    :attr:`num_groups`. The mean and standard-deviation are calculated\n    separately over the each group. :math:`\\gamma` and :math:`\\beta` are learnable\n    per-channel affine transform parameter vectors of size :attr:`num_channels` if\n    :attr:`affine` is ``True``.\n    The variance is calculated via the biased estimator, equivalent to\n    `torch.var(input, unbiased=False)`.\n\n    This layer uses statistics computed from input data in both training and\n    evaluation modes.\n\n    Args:\n        num_groups (int): number of groups to separate the channels into\n        num_channels (int): number of channels expected in input\n        eps: a value added to the denominator for numerical stability. Default: 1e-5\n        affine: a boolean value that when set to ``True``, this module\n            has learnable per-channel affine parameters initialized to ones (for weights)\n            and zeros (for biases). Default: ``True``.\n\n    Shape:\n        - Input: :math:`(N, C, *)` where :math:`C=\\text{num\\_channels}`\n        - Output: :math:`(N, C, *)` (same shape as input)\n\n    Examples::\n\n        >>> input = torch.randn(20, 6, 10, 10)\n        >>> # Separate 6 channels into 3 groups\n        >>> m = nn.GroupNorm(3, 6)\n        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)\n        >>> m = nn.GroupNorm(6, 6)\n        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)\n        >>> m = nn.GroupNorm(1, 6)\n        >>> # Activating the module\n        >>> output = m(input)\n    '
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float=1e-05, affine: bool=True, device=None, dtype=None, key=None, **kwargs) -> None:
        if key is None:
            key = jax.random.PRNGKey(0)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            key, subkey = jax.random.split(key)
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs), key=subkey)
            key, subkey = jax.random.split(key)
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs), key=subkey)
        else:
            jax.vmap(self.register_parameter)('weight', None)
            jax.vmap(self.register_parameter)('bias', None)
        jax.vmap(self.reset_parameters)()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def __call__(self, input: Tensor) -> Tensor:
        return F.group_norm(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, affine={affine}'.format(**self.__dict__)

class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    """Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.
        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where
          :math:`S=\\text{output\\_size}`.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5, 7))
        >>> input = torch.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = torch.randn(1, 64, 10, 9)
        >>> output = m(input)

    """
    output_size: _size_2_opt_t

    def forward(self, input: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input, self.output_size)

class LayerNorm(eqx.Module):
    normalized_shape: tuple
    weight: Parameter
    bias: Parameter
    'Applies Layer Normalization over a mini-batch of inputs.\n\n    This layer implements the operation as described in\n    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__\n\n    .. math::\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`\n    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`\n    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over\n    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).\n    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of\n    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.\n    The variance is calculated via the biased estimator, equivalent to\n    `torch.var(input, unbiased=False)`.\n\n    .. note::\n        Unlike Batch Normalization and Instance Normalization, which applies\n        scalar scale and bias for each entire channel/plane with the\n        :attr:`affine` option, Layer Normalization applies per-element scale and\n        bias with :attr:`elementwise_affine`.\n\n    This layer uses statistics computed from input data in both training and\n    evaluation modes.\n\n    Args:\n        normalized_shape (int or list or torch.Size): input shape from an expected input\n            of size\n\n            .. math::\n                [* \\times \\text{normalized\\_shape}[0] \\times \\text{normalized\\_shape}[1]\n                    \\times \\ldots \\times \\text{normalized\\_shape}[-1]]\n\n            If a single integer is used, it is treated as a singleton list, and this module will\n            normalize over the last dimension which is expected to be of that specific size.\n        eps: a value added to the denominator for numerical stability. Default: 1e-5\n        elementwise_affine: a boolean value that when set to ``True``, this module\n            has learnable per-element affine parameters initialized to ones (for weights)\n            and zeros (for biases). Default: ``True``.\n        bias: If set to ``False``, the layer will not learn an additive bias (only relevant if\n            :attr:`elementwise_affine` is ``True``). Default: ``True``.\n\n    Attributes:\n        weight: the learnable weights of the module of shape\n            :math:`\\text{normalized\\_shape}` when :attr:`elementwise_affine` is set to ``True``.\n            The values are initialized to 1.\n        bias:   the learnable bias of the module of shape\n                :math:`\\text{normalized\\_shape}` when :attr:`elementwise_affine` is set to ``True``.\n                The values are initialized to 0.\n\n    Shape:\n        - Input: :math:`(N, *)`\n        - Output: :math:`(N, *)` (same shape as input)\n\n    Examples::\n\n        >>> # NLP Example\n        >>> batch, sentence_length, embedding_dim = 20, 5, 10\n        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)\n        >>> layer_norm = nn.LayerNorm(embedding_dim)\n        >>> # Activate module\n        >>> layer_norm(embedding)\n        >>>\n        >>> # Image Example\n        >>> N, C, H, W = 20, 5, 10, 10\n        >>> input = torch.randn(N, C, H, W)\n        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)\n        >>> # as shown in the image below\n        >>> layer_norm = nn.LayerNorm([C, H, W])\n        >>> output = layer_norm(input)\n\n    .. image:: ../_static/img/nn/layer_norm.jpg\n        :scale: 50 %\n\n    '
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float=1e-05, elementwise_affine: bool=True, bias: bool=True, device=None, dtype=None, key=None, **kwargs) -> None:
        if key is None:
            key = jax.random.PRNGKey(0)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        key, subkey = jax.random.split(key)
        self.normalized_shape = tuple(normalized_shape, key=subkey)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            key, subkey = jax.random.split(key)
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs), key=subkey)
            if bias:
                key, subkey = jax.random.split(key)
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs), key=subkey)
            else:
                jax.vmap(self.register_parameter)('bias', None)
        else:
            jax.vmap(self.register_parameter)('weight', None)
            jax.vmap(self.register_parameter)('bias', None)
        jax.vmap(self.reset_parameters)()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def __call__(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class Sequential(eqx.Module):
    """A sequential container.

    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """
    _modules: dict[str, Module]

    @overload
    def __init__(self, key=None, *args: Module, **kwargs) -> None:
        if key is None:
            key = jax.random.PRNGKey(0)
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]', key=None, **kwargs) -> None:
        if key is None:
            key = jax.random.PRNGKey(0)
        ...

    def __init__(self, key=None, *args, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(0)
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                jax.vmap(self.add_module)(key, module)
        else:
            for idx, module in enumerate(args):
                jax.vmap(self.add_module)(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator."""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError(f'index {idx} is out of range')
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    @_copy_to_script_wrapper
    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError(f'add operator supports only objects of Sequential class, but {str(type(other))} is given.')

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> Self:
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError(f'add operator supports only objects of Sequential class, but {str(type(other))} is given.')

    def __mul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f'unsupported operand type(s) for *: {type(self)} and {type(other)}')
        elif other <= 0:
            raise ValueError(f'Non-positive multiplication factor {other} for {type(self)}')
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> 'Sequential':
        return self.__mul__(other)

    def __imul__(self, other: int) -> Self:
        if not isinstance(other, int):
            raise TypeError(f'unsupported operand type(s) for *: {type(self)} and {type(other)}')
        elif other <= 0:
            raise ValueError(f'Non-positive multiplication factor {other} for {type(self)}')
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __call__(self, input):
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        """Append a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise AssertionError(f'module should be of type: {Module}')
        n = len(self._modules)
        if not -n <= index <= n:
            raise IndexError(f'Index out of range: {index}')
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> 'Sequential':
        for layer in sequential:
            self.append(layer)
        return self