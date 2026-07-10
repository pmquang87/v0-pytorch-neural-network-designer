# PyTorch Layer Reference

This document describes how the layers available in the **PyTorch Neural Network
Designer** map onto the [`torch.nn`](https://docs.pytorch.org/docs/stable/nn.html)
API, which constructor arguments the code generator emits, and where to find the
authoritative PyTorch documentation for each one.

It is the human-readable companion to the machine-readable
[`PYTORCH_LAYER_MANIFEST`](../lib/types.ts), which is the single source of truth
the code generator uses. When you add or change a layer there, update this file to
match.

## PyTorch version compatibility

The generated code targets the **PyTorch 2.x** stable API. As of mid-2026 the
current stable release is **PyTorch 2.11** (Python 3.10–3.14), and every module
listed here is available in the `torch.nn` namespace. A few layers were added in
recent releases:

| Layer | Available since | Notes |
| --- | --- | --- |
| `nn.RMSNorm` | PyTorch **2.4** | Root-mean-square layer norm used by LLaMA, Gemma, and most modern LLMs. Requires PyTorch ≥ 2.4. |
| `nn.GELU(approximate=...)` | PyTorch **1.12** | The `approximate` argument (`'none'` / `'tanh'`) selects the exact or tanh-approximated formulation. |
| `nn.Mish`, `nn.Hardswish`, `nn.SiLU` | PyTorch ≥ 1.7 | Modern activations widely used in EfficientNet / MobileNetV3-style networks. |

If you need to target an older PyTorch, avoid the layers noted above (in
particular `nn.RMSNorm`, which will raise `AttributeError` on PyTorch < 2.4).

## How code generation works

Each node on the canvas has a `type` (for example `conv2dNode`). The generator
looks the type up in `PYTORCH_LAYER_MANIFEST` to find:

- **`className`** – the `torch.nn` class to instantiate (or `null` for nodes that
  are handled inline in the `forward` pass, such as `Add` or `Concatenate`).
- **`params`** – the constructor arguments to emit, **in order**. Only arguments
  that are actually set on the node are written, so PyTorch defaults apply to the
  rest.
- **`doc`** – a link to the official documentation page for that layer.

For example, a `linearNode` with `in_features=784`, `out_features=128` produces:

```python
self.layer = nn.Linear(in_features=784, out_features=128)
```

Values are converted to Python literals: booleans become `True`/`False`, arrays
become tuples, and numeric strings are left unquoted.

## Supported layers

The tables below group every supported node by category. The **Node type** column
is the internal identifier; the **PyTorch class** is what the generator emits.

### Linear

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `linearNode` | `nn.Linear` | `in_features`, `out_features`, `bias` | [Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) |

### Convolution

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `conv1dNode` | `nn.Conv1d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias` | [Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) |
| `conv2dNode` | `nn.Conv2d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias` | [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |
| `conv3dNode` | `nn.Conv3d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`, `bias` | [Conv3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html) |
| `depthwiseconv2dNode` | `nn.Conv2d` (with `groups=in_channels`) | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `bias` | [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |
| `separableconv2dNode` | `nn.Sequential` (depthwise + pointwise `nn.Conv2d`) | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding` | [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) |

### Transposed convolution

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `convtranspose1dNode` | `nn.ConvTranspose1d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `output_padding`, `groups`, `bias`, `dilation` | [ConvTranspose1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html) |
| `convtranspose2dNode` | `nn.ConvTranspose2d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `output_padding`, `groups`, `bias`, `dilation` | [ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html) |
| `convtranspose3dNode` | `nn.ConvTranspose3d` | `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `output_padding`, `groups`, `bias`, `dilation` | [ConvTranspose3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html) |

### Pooling

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `maxpool1dNode` | `nn.MaxPool1d` | `kernel_size`, `stride`, `padding`, `dilation`, `return_indices`, `ceil_mode` | [MaxPool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html) |
| `maxpool2dNode` | `nn.MaxPool2d` | `kernel_size`, `stride`, `padding`, `dilation`, `return_indices`, `ceil_mode` | [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) |
| `maxpool3dNode` | `nn.MaxPool3d` | `kernel_size`, `stride`, `padding`, `dilation`, `return_indices`, `ceil_mode` | [MaxPool3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool3d.html) |
| `avgpool1dNode` | `nn.AvgPool1d` | `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad` | [AvgPool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html) |
| `avgpool2dNode` | `nn.AvgPool2d` | `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad`, `divisor_override` | [AvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html) |
| `avgpool3dNode` | `nn.AvgPool3d` | `kernel_size`, `stride`, `padding`, `ceil_mode`, `count_include_pad`, `divisor_override` | [AvgPool3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AvgPool3d.html) |
| `adaptiveavgpool1dNode` | `nn.AdaptiveAvgPool1d` | `output_size` | [AdaptiveAvgPool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool1d.html) |
| `adaptiveavgpool2dNode` | `nn.AdaptiveAvgPool2d` | `output_size` | [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html) |
| `adaptiveavgpool3dNode` | `nn.AdaptiveAvgPool3d` | `output_size` | [AdaptiveAvgPool3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool3d.html) |
| `adaptivemaxpool1dNode` | `nn.AdaptiveMaxPool1d` | `output_size` | [AdaptiveMaxPool1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool1d.html) |
| `adaptivemaxpool2dNode` | `nn.AdaptiveMaxPool2d` | `output_size` | [AdaptiveMaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html) |
| `adaptivemaxpool3dNode` | `nn.AdaptiveMaxPool3d` | `output_size` | [AdaptiveMaxPool3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool3d.html) |
| `lppool2dNode` | `nn.LPPool2d` | `norm_type`, `kernel_size`, `stride`, `ceil_mode` | [LPPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.LPPool2d.html) |
| `fractionalmaxpool2dNode` | `nn.FractionalMaxPool2d` | `kernel_size`, `output_size`, `output_ratio`, `return_indices` | [FractionalMaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.FractionalMaxPool2d.html) |

### Normalization

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `batchnorm1dNode` | `nn.BatchNorm1d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html) |
| `batchnorm2dNode` | `nn.BatchNorm2d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) |
| `batchnorm3dNode` | `nn.BatchNorm3d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [BatchNorm3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm3d.html) |
| `layernormNode` | `nn.LayerNorm` | `normalized_shape`, `eps`, `elementwise_affine` | [LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) |
| `rmsnormNode` | `nn.RMSNorm` | `normalized_shape`, `eps`, `elementwise_affine` | [RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) |
| `instancenorm1dNode` | `nn.InstanceNorm1d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [InstanceNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm1d.html) |
| `instancenorm2dNode` | `nn.InstanceNorm2d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [InstanceNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html) |
| `instancenorm3dNode` | `nn.InstanceNorm3d` | `num_features`, `eps`, `momentum`, `affine`, `track_running_stats` | [InstanceNorm3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.InstanceNorm3d.html) |
| `groupnormNode` | `nn.GroupNorm` | `num_groups`, `num_channels`, `eps`, `affine` | [GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html) |

> **RMSNorm vs. LayerNorm.** `nn.RMSNorm` normalizes by the root-mean-square of the
> activations *without* subtracting the mean, and (with the default
> `elementwise_affine=True`) has a single learnable scale and **no bias**. This
> makes it cheaper than `nn.LayerNorm` and is the norm of choice in most modern
> transformer LLMs.

### Activation

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `reluNode` | `nn.ReLU` | `inplace` | [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html) |
| `leakyreluNode` | `nn.LeakyReLU` | `negative_slope`, `inplace` | [LeakyReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html) |
| `eluNode` | `nn.ELU` | `alpha`, `inplace` | [ELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ELU.html) |
| `seluNode` | `nn.SELU` | `inplace` | [SELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SELU.html) |
| `geluNode` | `nn.GELU` | `approximate` (`'none'` \| `'tanh'`) | [GELU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html) |
| `siluNode` | `nn.SiLU` | `inplace` | [SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html) |
| `mishNode` | `nn.Mish` | `inplace` | [Mish](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html) |
| `hardswishNode` | `nn.Hardswish` | `inplace` | [Hardswish](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardswish.html) |
| `hardsigmoidNode` | `nn.Hardsigmoid` | `inplace` | [Hardsigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html) |
| `sigmoidNode` | `nn.Sigmoid` | – | [Sigmoid](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) |
| `tanhNode` | `nn.Tanh` | – | [Tanh](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html) |
| `softmaxNode` | `nn.Softmax` | `dim` | [Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) |
| `logsoftmaxNode` | `nn.LogSoftmax` | `dim` | [LogSoftmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html) |

> **GELU `approximate`.** The default `'none'` computes the exact GELU using the
> Gaussian CDF. `'tanh'` uses the faster tanh approximation
> `0.5 · x · (1 + tanh(√(2/π)·(x + 0.044715·x³)))`. The designer only emits the
> argument when you change it away from the default.

### Regularization

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `dropoutNode` | `nn.Dropout` | `p`, `inplace` | [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) |
| `dropout2dNode` | `nn.Dropout2d` | `p`, `inplace` | [Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html) |
| `dropout3dNode` | `nn.Dropout3d` | `p`, `inplace` | [Dropout3d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout3d.html) |

### Recurrent

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `lstmNode` | `nn.LSTM` | `input_size`, `hidden_size`, `num_layers`, `bias`, `batch_first`, `dropout`, `bidirectional` | [LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html) |
| `gruNode` | `nn.GRU` | `input_size`, `hidden_size`, `num_layers`, `bias`, `batch_first`, `dropout`, `bidirectional` | [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html) |
| `rnnNode` | `nn.RNN` | `input_size`, `hidden_size`, `num_layers`, `nonlinearity`, `bias`, `batch_first`, `dropout`, `bidirectional` | [RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) |

> Recurrent modules return a `(output, hidden_state)` tuple; the generator unpacks
> the first element (`output, _ = self.layer(x)`) so downstream nodes receive the
> sequence output.

### Attention & Transformers

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `multiheadattentionNode` | `nn.MultiheadAttention` | `embed_dim`, `num_heads`, `dropout`, `bias`, `add_bias_kv`, `add_zero_attn`, `kdim`, `vdim` | [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) |
| `transformerencoderlayerNode` | `nn.TransformerEncoderLayer` | `d_model`, `nhead`, `dim_feedforward`, `dropout`, `activation`, `layer_norm_eps`, `batch_first`, `norm_first` | [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) |
| `transformerdecoderlayerNode` | `nn.TransformerDecoderLayer` | `d_model`, `nhead`, `dim_feedforward`, `dropout`, `activation`, `layer_norm_eps`, `batch_first`, `norm_first` | [TransformerDecoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html) |

> `nn.MultiheadAttention` takes `(query, key, value)` and returns
> `(attn_output, attn_weights)`. With a single incoming edge the generator wires it
> as self-attention (`q = k = v`) and keeps only `attn_output`. For the raw
> attention primitive, see
> [`torch.nn.functional.scaled_dot_product_attention`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

### Shape / utility

| Node type | PyTorch class | Parameters | Docs |
| --- | --- | --- | --- |
| `flattenNode` | `nn.Flatten` | `start_dim`, `end_dim` | [Flatten](https://docs.pytorch.org/docs/stable/generated/torch.nn.Flatten.html) |
| `upsampleNode` | `nn.Upsample` | `size`, `scale_factor`, `mode`, `align_corners` | [Upsample](https://docs.pytorch.org/docs/stable/generated/torch.nn.Upsample.html) |
| `reshapeNode` | `Tensor.view` (inline) | `targetShape` | [Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) |
| `transposeNode` | `torch.transpose` (inline) | `dim0`, `dim1` | [torch.transpose](https://docs.pytorch.org/docs/stable/generated/torch.transpose.html) |

### Combining operations (handled inline in `forward`)

These nodes do not create an `nn.Module`; they are emitted directly in the
`forward` pass.

| Node type | Emitted as | Parameters | Docs |
| --- | --- | --- | --- |
| `addNode` | `a + b` (elementwise, e.g. residual skip connection) | – | – |
| `multiplyNode` | `a * b` (elementwise) | – | – |
| `concatenateNode` | `torch.cat([...], dim=...)` | `dim` | [torch.cat](https://docs.pytorch.org/docs/stable/generated/torch.cat.html) |

### Composite / block layers

Higher-level building blocks composed from the primitives above. These are
generated with dedicated logic rather than a single `nn` class.

| Node type | Description |
| --- | --- |
| `mbconvNode` | Mobile inverted bottleneck (MBConv) block used in MobileNetV2/V3 and EfficientNet. |
| `invertedResidualBlockNode` | Inverted residual block (MobileNetV2). |
| `seBlockNode` / `seBottleneckNode` | Squeeze-and-Excitation channel-attention block / bottleneck. |
| `ssmNode` | Selective state-space model (Mamba-style) block. |

## Further reading

- [`torch.nn` — full module index](https://docs.pytorch.org/docs/stable/nn.html)
- [PyTorch documentation home](https://docs.pytorch.org/docs/stable/index.html)
- [PyTorch release notes](https://github.com/pytorch/pytorch/releases)
