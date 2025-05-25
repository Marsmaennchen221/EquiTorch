# PyTorch to Equinox Module Mapping

This table provides a comprehensive mapping of PyTorch builtin modules and networks to their Equinox equivalents.

## Neural Network Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☑️ | `torch.nn.Linear` | `equinox.nn.Linear` | Direct equivalent with `use_bias` parameter |
| ☑️ | `torch.nn.Conv1d` | `equinox.nn.Conv1d` | Direct equivalent |
| ☑️ | `torch.nn.Conv2d` | `equinox.nn.Conv2d` | Direct equivalent |
| ☑️ | `torch.nn.Conv3d` | `equinox.nn.Conv3d` | Direct equivalent |
| ☐ | `torch.nn.ConvTranspose1d` | `equinox.nn.ConvTranspose1d` | Direct equivalent |
| ☐ | `torch.nn.ConvTranspose2d` | `equinox.nn.ConvTranspose2d` | Direct equivalent |
| ☐ | `torch.nn.ConvTranspose3d` | `equinox.nn.ConvTranspose3d` | Direct equivalent |

## Activation Functions

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☑️ | `torch.nn.ReLU` | `jax.nn.relu` | Use as function, not module |
| ☐ | `torch.nn.LeakyReLU` | `jax.nn.leaky_relu` | Use as function, not module |
| ☐ | `torch.nn.ELU` | `jax.nn.elu` | Use as function, not module |
| ☐ | `torch.nn.GELU` | `jax.nn.gelu` | Use as function, not module |
| ☐ | `torch.nn.Sigmoid` | `jax.nn.sigmoid` | Use as function, not module |
| ☐ | `torch.nn.Tanh` | `jax.nn.tanh` | Use as function, not module |
| ☐ | `torch.nn.Softmax` | `jax.nn.softmax` | Use as function, not module |
| ☐ | `torch.nn.LogSoftmax` | `jax.nn.log_softmax` | Use as function, not module |
| ☐ | `torch.nn.Swish` | `jax.nn.swish` | Use as function, not module |
| ☐ | `torch.nn.SiLU` | `jax.nn.silu` | Use as function, not module |
| ☐ | `torch.nn.Mish` | `jax.nn.mish` | Use as function, not module |
| ☐ | `torch.nn.PReLU` | `equinox.nn.PReLU` | Direct equivalent |
| ☐ | `torch.nn.ReLU6` | `jax.nn.relu6` | Use as function, not module |
| ☐ | `torch.nn.Hardtanh` | `jax.nn.hard_tanh` | Use as function, not module |
| ☐ | `torch.nn.Hardsigmoid` | `jax.nn.hard_sigmoid` | Use as function, not module |
| ☐ | `torch.nn.Hardswish` | `jax.nn.hard_swish` | Use as function, not module |

## Normalization Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.BatchNorm1d` | `equinox.nn.BatchNorm` | Direct equivalent |
| ☐ | `torch.nn.BatchNorm2d` | `equinox.nn.BatchNorm` | Direct equivalent |
| ☐ | `torch.nn.BatchNorm3d` | `equinox.nn.BatchNorm` | Direct equivalent |
| ☐ | `torch.nn.LayerNorm` | `equinox.nn.LayerNorm` | Direct equivalent |
| ☐ | `torch.nn.GroupNorm` | `equinox.nn.GroupNorm` | Direct equivalent |
| ☐ | `torch.nn.InstanceNorm1d` | Manual implementation | Use `jax.nn.normalize` with appropriate axes |
| ☐ | `torch.nn.InstanceNorm2d` | Manual implementation | Use `jax.nn.normalize` with appropriate axes |
| ☐ | `torch.nn.InstanceNorm3d` | Manual implementation | Use `jax.nn.normalize` with appropriate axes |
| ☐ | `torch.nn.LocalResponseNorm` | Manual implementation | No direct equivalent |

## Recurrent Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.RNN` | `equinox.nn.RNNCell` + `jax.lax.scan` | Use cell with scan for sequences |
| ☐ | `torch.nn.LSTM` | `equinox.nn.LSTMCell` + `jax.lax.scan` | Use cell with scan for sequences |
| ☐ | `torch.nn.GRU` | `equinox.nn.GRUCell` + `jax.lax.scan` | Use cell with scan for sequences |
| ☐ | `torch.nn.RNNCell` | `equinox.nn.RNNCell` | Direct equivalent |
| ☐ | `torch.nn.LSTMCell` | `equinox.nn.LSTMCell` | Direct equivalent |
| ☐ | `torch.nn.GRUCell` | `equinox.nn.GRUCell` | Direct equivalent |

## Transformer Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Transformer` | `equinox.nn.MultiheadAttention` + custom | Build using attention blocks |
| ☐ | `torch.nn.TransformerEncoder` | Custom implementation | Combine multiple encoder layers |
| ☐ | `torch.nn.TransformerDecoder` | Custom implementation | Combine multiple decoder layers |
| ☐ | `torch.nn.TransformerEncoderLayer` | Custom implementation | Use `MultiheadAttention` + `Linear` + `LayerNorm` |
| ☐ | `torch.nn.TransformerDecoderLayer` | Custom implementation | Use `MultiheadAttention` + `Linear` + `LayerNorm` |
| ☐ | `torch.nn.MultiheadAttention` | `equinox.nn.MultiheadAttention` | Direct equivalent |

## Embedding Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Embedding` | `equinox.nn.Embedding` | Direct equivalent |
| ☐ | `torch.nn.EmbeddingBag` | Manual implementation | No direct equivalent |

## Pooling Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☑️ | `torch.nn.MaxPool1d` | `jax.lax.reduce_window` | Use with `jax.lax.max` |
| ☑️ | `torch.nn.MaxPool2d` | `jax.lax.reduce_window` | Use with `jax.lax.max` |
| ☑️ | `torch.nn.MaxPool3d` | `jax.lax.reduce_window` | Use with `jax.lax.max` |
| ☐ | `torch.nn.AvgPool1d` | `jax.lax.reduce_window` | Use with mean reduction |
| ☐ | `torch.nn.AvgPool2d` | `jax.lax.reduce_window` | Use with mean reduction |
| ☐ | `torch.nn.AvgPool3d` | `jax.lax.reduce_window` | Use with mean reduction |
| ☐ | `torch.nn.AdaptiveMaxPool1d` | `equinox.nn.AdaptivePool` | Direct equivalent |
| ☐ | `torch.nn.AdaptiveMaxPool2d` | `equinox.nn.AdaptivePool` | Direct equivalent |
| ☐ | `torch.nn.AdaptiveMaxPool3d` | `equinox.nn.AdaptivePool` | Direct equivalent |
| ☐ | `torch.nn.AdaptiveAvgPool1d` | `equinox.nn.AdaptiveAvgPool` | Direct equivalent |
| ☐ | `torch.nn.AdaptiveAvgPool2d` | `equinox.nn.AdaptiveAvgPool` | Direct equivalent |
| ☐ | `torch.nn.AdaptiveAvgPool3d` | `equinox.nn.AdaptiveAvgPool` | Direct equivalent |
| ☐ | `torch.nn.GlobalAvgPool2d` | `jnp.mean` with axis | Use `jnp.mean(x, axis=(2, 3))` |

## Regularization Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Dropout` | `equinox.nn.Dropout` | Direct equivalent |
| ☐ | `torch.nn.Dropout1d` | `equinox.nn.Dropout` | Use with appropriate shape |
| ☐ | `torch.nn.Dropout2d` | `equinox.nn.Dropout` | Use with appropriate shape |
| ☐ | `torch.nn.Dropout3d` | `equinox.nn.Dropout` | Use with appropriate shape |
| ☐ | `torch.nn.AlphaDropout` | Manual implementation | No direct equivalent |
| ☐ | `torch.nn.FeatureAlphaDropout` | Manual implementation | No direct equivalent |

## Loss Functions

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.MSELoss` | `optax.l2_loss` | Use Optax loss functions |
| ☐ | `torch.nn.CrossEntropyLoss` | `optax.softmax_cross_entropy` | Use Optax loss functions |
| ☐ | `torch.nn.BCELoss` | `optax.sigmoid_binary_cross_entropy` | Use Optax loss functions |
| ☐ | `torch.nn.BCEWithLogitsLoss` | `optax.sigmoid_binary_cross_entropy` | Use Optax loss functions |
| ☐ | `torch.nn.L1Loss` | `optax.l1_loss` | Use Optax loss functions |
| ☐ | `torch.nn.SmoothL1Loss` | `optax.smooth_labels` + `optax.l1_loss` | Combine Optax functions |
| ☐ | `torch.nn.NLLLoss` | `optax.softmax_cross_entropy` | Use Optax loss functions |
| ☐ | `torch.nn.KLDivLoss` | Manual implementation | Use `jnp.sum(p * jnp.log(p / q))` |
| ☐ | `torch.nn.HuberLoss` | `optax.huber_loss` | Use Optax loss functions |

## Padding Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.ReflectionPad1d` | `jnp.pad` | Use with 'reflect' mode |
| ☐ | `torch.nn.ReflectionPad2d` | `jnp.pad` | Use with 'reflect' mode |
| ☐ | `torch.nn.ReplicationPad1d` | `jnp.pad` | Use with 'edge' mode |
| ☐ | `torch.nn.ReplicationPad2d` | `jnp.pad` | Use with 'edge' mode |
| ☐ | `torch.nn.ReplicationPad3d` | `jnp.pad` | Use with 'edge' mode |
| ☐ | `torch.nn.ZeroPad2d` | `jnp.pad` | Use with 'constant' mode |
| ☐ | `torch.nn.ConstantPad1d` | `jnp.pad` | Use with 'constant' mode |
| ☐ | `torch.nn.ConstantPad2d` | `jnp.pad` | Use with 'constant' mode |
| ☐ | `torch.nn.ConstantPad3d` | `jnp.pad` | Use with 'constant' mode |

## Upsampling Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Upsample` | `jax.image.resize` | Use JAX image resize functions |
| ☐ | `torch.nn.UpsamplingNearest2d` | `jax.image.resize` | Use with 'nearest' method |
| ☐ | `torch.nn.UpsamplingBilinear2d` | `jax.image.resize` | Use with 'bilinear' method |

## Vision Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.PixelShuffle` | Manual implementation | Use `jnp.reshape` and `jnp.transpose` |
| ☐ | `torch.nn.PixelUnshuffle` | Manual implementation | Use `jnp.reshape` and `jnp.transpose` |

## Sparse Layers

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Embedding` | `equinox.nn.Embedding` | Direct equivalent |
| ☐ | `torch.nn.EmbeddingBag` | Manual implementation | No direct equivalent |

## Distance Functions

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.CosineSimilarity` | Manual implementation | Use `jnp.dot` and `jnp.linalg.norm` |
| ☐ | `torch.nn.PairwiseDistance` | Manual implementation | Use `jnp.linalg.norm` |

## Container Modules

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Sequential` | `equinox.nn.Sequential` | Direct equivalent |
| ☐ | `torch.nn.ModuleList` | Python list | Use regular Python lists |
| ☐ | `torch.nn.ModuleDict` | Python dict | Use regular Python dictionaries |
| ☐ | `torch.nn.ParameterList` | Python list | Use regular Python lists |
| ☐ | `torch.nn.ParameterDict` | Python dict | Use regular Python dictionaries |

## Utilities

| ☐ | PyTorch Module | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torch.nn.Identity` | `equinox.nn.Identity` | Direct equivalent |
| ☐ | `torch.nn.Flatten` | `jnp.reshape` | Use `x.reshape(x.shape[0], -1)` |
| ☐ | `torch.nn.Unflatten` | `jnp.reshape` | Use with target shape |

## Functional Equivalents

| ☐ | PyTorch Function | JAX/Equinox Equivalent | Notes |
|---|---|---|---|
| ☑️ | `torch.nn.functional.relu` | `jax.nn.relu` | Direct equivalent |
| ☐ | `torch.nn.functional.gelu` | `jax.nn.gelu` | Direct equivalent |
| ☐ | `torch.nn.functional.softmax` | `jax.nn.softmax` | Direct equivalent |
| ☐ | `torch.nn.functional.log_softmax` | `jax.nn.log_softmax` | Direct equivalent |
| ☐ | `torch.nn.functional.dropout` | `equinox.nn.Dropout` | Use as module |
| ☐ | `torch.nn.functional.batch_norm` | `equinox.nn.BatchNorm` | Use as module |
| ☐ | `torch.nn.functional.layer_norm` | `equinox.nn.LayerNorm` | Use as module |
| ☐ | `torch.nn.functional.conv1d` | `jax.lax.conv_general_dilated` | Lower-level implementation |
| ☐ | `torch.nn.functional.conv2d` | `jax.lax.conv_general_dilated` | Lower-level implementation |
| ☐ | `torch.nn.functional.conv3d` | `jax.lax.conv_general_dilated` | Lower-level implementation |
| ☑️ | `torch.nn.functional.max_pool1d` | `jax.lax.reduce_window` | Use with max operation |
| ☑️ | `torch.nn.functional.max_pool2d` | `jax.lax.reduce_window` | Use with max operation |
| ☑️ | `torch.nn.functional.max_pool3d` | `jax.lax.reduce_window` | Use with max operation |
| ☐ | `torch.nn.functional.avg_pool1d` | `jax.lax.reduce_window` | Use with mean operation |
| ☐ | `torch.nn.functional.avg_pool2d` | `jax.lax.reduce_window` | Use with mean operation |
| ☐ | `torch.nn.functional.interpolate` | `jax.image.resize` | Direct equivalent |
| ☐ | `torch.nn.functional.grid_sample` | Manual implementation | No direct equivalent |
| ☐ | `torch.nn.functional.affine_grid` | Manual implementation | No direct equivalent |

## Pre-trained Models and Architectures

| ☐ | PyTorch Model | Equinox Equivalent | Notes |
|---|---|---|---|
| ☐ | `torchvision.models.resnet*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.vgg*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.alexnet` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.densenet*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.inception*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.mobilenet*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.efficientnet*` | Manual implementation | Implement using Equinox modules |
| ☐ | `torchvision.models.vision_transformer` | Manual implementation | Use `equinox.nn.MultiheadAttention` |

## Key Differences and Migration Notes

### 1. **Functional vs Module-based Activations**
- PyTorch: Can use both `nn.ReLU()` (module) and `F.relu()` (function)
- Equinox: Prefer JAX functions like `jax.nn.relu()` for activations

### 2. **Random Number Generation**
- PyTorch: Global random state
- Equinox: Explicit PRNG keys required for all random operations

### 3. **Batch Dimensions**
- PyTorch: Implicit batch handling
- Equinox: Explicit batch handling with `jax.vmap`

### 4. **Training vs Inference Mode**
- PyTorch: `model.train()` and `model.eval()`
- Equinox: Pass explicit flags to modules (e.g., `training=True` for BatchNorm)

### 5. **Parameter Initialization**
- PyTorch: Automatic initialization
- Equinox: Explicit key-based initialization

### 6. **Loss Functions**
- PyTorch: Built into `torch.nn`
- Equinox: Use Optax library for loss functions

### 7. **Optimizers**
- PyTorch: Built into `torch.optim`
- Equinox: Use Optax library for optimizers

### 8. **Device Management**
- PyTorch: Explicit `.to(device)` calls
- JAX/Equinox: Automatic device placement, use `jax.device_put` if needed
