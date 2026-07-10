// Core types for the neural network designer

export interface TensorShape {
  batch: number | "dynamic"
  channels?: number | "dynamic"
  height?: number | "dynamic"
  width?: number | "dynamic"
  depth?: number | "dynamic"
  features?: number | "dynamic"
  sequence?: number | "dynamic"
  length?: number | "dynamic"
}

// Base node data interface
export interface BaseNodeData {
  name?: string
  inputShape?: TensorShape
  outputShape?: TensorShape
  [key: string]: any
}

// Input node data
export interface InputNodeData extends BaseNodeData {
  batch_size: number
  channels?: number
  height?: number
  width?: number
  features?: number
}

// Linear layer data
export interface LinearNodeData extends BaseNodeData {
  in_features: number
  out_features: number
}

// Convolutional layer data
export interface ConvNodeData extends BaseNodeData {
  in_channels: number
  out_channels: number
  kernel_size: number | [number, number] | [number, number, number]
  stride?: number | [number, number] | [number, number, number]
  padding?: number | [number, number] | [number, number, number]
  dilation?: number | [number, number] | [number, number, number]
  groups?: number
  bias?: boolean
}

// Pooling layer data
export interface PoolNodeData extends BaseNodeData {
  kernel_size: number | [number, number] | [number, number, number]
  stride?: number | [number, number] | [number, number, number]
  padding?: number | [number, number] | [number, number, number]
  dilation?: number | [number, number] | [number, number, number]
  return_indices?: boolean
  ceil_mode?: boolean
}

// Normalization layer data
export interface NormNodeData extends BaseNodeData {
  num_features?: number
  eps?: number
  momentum?: number
  affine?: boolean
  track_running_stats?: boolean
  num_groups?: number
  num_channels?: number
}

// Activation function data
export interface ActivationNodeData extends BaseNodeData {
  inplace?: boolean
  negative_slope?: number // for LeakyReLU
  alpha?: number // for ELU
  beta?: number // for Hardswish
}

// Dropout data
export interface DropoutNodeData extends BaseNodeData {
  p: number
  inplace?: boolean
}

// LSTM/GRU/RNN data
export interface RNNNodeData extends BaseNodeData {
  input_size: number
  hidden_size: number
  num_layers?: number
  bias?: boolean
  batch_first?: boolean
  dropout?: number
  bidirectional?: boolean
  nonlinearity?: string // for RNN
}

// Multihead attention data
export interface AttentionNodeData extends BaseNodeData {
  embed_dim: number
  num_heads: number
  dropout?: number
  bias?: boolean
  add_bias_kv?: boolean
  add_zero_attn?: boolean
  kdim?: number
  vdim?: number
}

// Transformer layer data
export interface TransformerNodeData extends BaseNodeData {
  d_model: number
  nhead: number
  dim_feedforward?: number
  dropout?: number
  activation?: string
  layer_norm_eps?: number
  batch_first?: boolean
  norm_first?: boolean
}

// Operation node data
export interface OperationNodeData extends BaseNodeData {
  dim?: number
  index?: number
  start_dim?: number
  end_dim?: number
}

// MBConv node data
export interface MBConvNodeData extends BaseNodeData {
  in_channels: number;
  out_channels: number;
  kernel_size: number;
  stride: number;
  expand_ratio: number;
  se_ratio?: number;
}

// Mixture-of-Experts (sparse MoE feed-forward) node data
export interface MoENodeData extends BaseNodeData {
  d_model: number;      // token / model dimension (in == out)
  d_ff: number;         // hidden dimension of each expert FFN
  num_experts: number;  // total number of experts
  top_k: number;        // experts activated per token
  activation?: string;  // "gelu" | "silu" | "relu"
}

// Union type for all node data
export type NodeData = 
  | InputNodeData
  | LinearNodeData
  | ConvNodeData
  | PoolNodeData
  | NormNodeData
  | ActivationNodeData
  | DropoutNodeData
  | RNNNodeData
  | AttentionNodeData
  | TransformerNodeData
  | OperationNodeData
  | MBConvNodeData
  | MoENodeData

// Network state for undo/redo
export interface NetworkState {
  nodes: any[]
  edges: any[]
  timestamp: number
}

// Model validation result
export interface ValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
}

// Keyboard shortcut configuration
export interface KeyboardShortcut {
  key: string
  ctrlKey?: boolean
  shiftKey?: boolean
  altKey?: boolean
  action: () => void
  description: string
}

// Help content structure
export interface HelpContent {
  title: string
  description: string
  parameters?: Array<{
    name: string
    type: string
    description: string
    default?: any
  }>
  examples?: string[]
  tips?: string[]
}

// Graph IR representation
export interface GraphIR {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GraphNode {
  id: string
  type: string
  data: NodeData
  position?: { x: number; y: number }
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  type?: string
  targetHandle?: string
}

// API Request/Response types
export interface GenerateModelRequest {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GenerateModelResponse {
  success: boolean
  code?: string
  error?: string
}

// Layer manifest entry describing how a canvas node maps onto a PyTorch module.
// `className` is the `torch.nn` class emitted by the code generator (or `null`
// for nodes handled specially / inline in the forward pass). `params` lists the
// constructor arguments emitted in declaration order. `doc` links to the
// matching page in the official PyTorch documentation (stable channel).
export interface LayerManifestEntry {
  className: string | null
  params: string[]
  doc: string | null
}

// Base URL for the stable PyTorch documentation. Individual layer pages follow
// the `generated/torch.nn.<Class>.html` convention.
export const PYTORCH_DOCS_BASE = "https://docs.pytorch.org/docs/stable"

const nnDoc = (cls: string): string => `${PYTORCH_DOCS_BASE}/generated/torch.nn.${cls}.html`

// PyTorch layer manifest - defines layer parameters, class names, and doc links.
// Verified against the PyTorch 2.x `torch.nn` API (stable docs).
export const PYTORCH_LAYER_MANIFEST: Record<string, LayerManifestEntry> = {
  linearNode: {
    className: "nn.Linear",
    params: ["in_features", "out_features", "bias"],
    doc: nnDoc("Linear")
  },
  conv1dNode: {
    className: "nn.Conv1d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
    doc: nnDoc("Conv1d")
  },
  conv2dNode: {
    className: "nn.Conv2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
    doc: nnDoc("Conv2d")
  },
  conv3dNode: {
    className: "nn.Conv3d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
    doc: nnDoc("Conv3d")
  },
  maxpool1dNode: {
    className: "nn.MaxPool1d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
    doc: nnDoc("MaxPool1d")
  },
  maxpool2dNode: {
    className: "nn.MaxPool2d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
    doc: nnDoc("MaxPool2d")
  },
  maxpool3dNode: {
    className: "nn.MaxPool3d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"],
    doc: nnDoc("MaxPool3d")
  },
  avgpool1dNode: {
    className: "nn.AvgPool1d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"],
    doc: nnDoc("AvgPool1d")
  },
  avgpool2dNode: {
    className: "nn.AvgPool2d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
    doc: nnDoc("AvgPool2d")
  },
  avgpool3dNode: {
    className: "nn.AvgPool3d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"],
    doc: nnDoc("AvgPool3d")
  },
  adaptiveavgpool1dNode: {
    className: "nn.AdaptiveAvgPool1d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveAvgPool1d")
  },
  adaptiveavgpool2dNode: {
    className: "nn.AdaptiveAvgPool2d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveAvgPool2d")
  },
  adaptiveavgpool3dNode: {
    className: "nn.AdaptiveAvgPool3d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveAvgPool3d")
  },
  adaptivemaxpool1dNode: {
    className: "nn.AdaptiveMaxPool1d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveMaxPool1d")
  },
  adaptivemaxpool2dNode: {
    className: "nn.AdaptiveMaxPool2d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveMaxPool2d")
  },
  adaptivemaxpool3dNode: {
    className: "nn.AdaptiveMaxPool3d",
    params: ["output_size"],
    doc: nnDoc("AdaptiveMaxPool3d")
  },
  lppool2dNode: {
    className: "nn.LPPool2d",
    params: ["norm_type", "kernel_size", "stride", "ceil_mode"],
    doc: nnDoc("LPPool2d")
  },
  fractionalmaxpool2dNode: {
    className: "nn.FractionalMaxPool2d",
    params: ["kernel_size", "output_size", "output_ratio", "return_indices"],
    doc: nnDoc("FractionalMaxPool2d")
  },
  batchnorm1dNode: {
    className: "nn.BatchNorm1d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("BatchNorm1d")
  },
  batchnorm2dNode: {
    className: "nn.BatchNorm2d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("BatchNorm2d")
  },
  batchnorm3dNode: {
    className: "nn.BatchNorm3d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("BatchNorm3d")
  },
  layernormNode: {
    className: "nn.LayerNorm",
    params: ["normalized_shape", "eps", "elementwise_affine"],
    doc: nnDoc("LayerNorm")
  },
  rmsnormNode: {
    className: "nn.RMSNorm",
    params: ["normalized_shape", "eps", "elementwise_affine"],
    doc: nnDoc("RMSNorm")
  },
  instancenorm1dNode: {
    className: "nn.InstanceNorm1d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("InstanceNorm1d")
  },
  instancenorm2dNode: {
    className: "nn.InstanceNorm2d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("InstanceNorm2d")
  },
  instancenorm3dNode: {
    className: "nn.InstanceNorm3d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"],
    doc: nnDoc("InstanceNorm3d")
  },
  groupnormNode: {
    className: "nn.GroupNorm",
    params: ["num_groups", "num_channels", "eps", "affine"],
    doc: nnDoc("GroupNorm")
  },
  reluNode: {
    className: "nn.ReLU",
    params: ["inplace"],
    doc: nnDoc("ReLU")
  },
  leakyreluNode: {
    className: "nn.LeakyReLU",
    params: ["negative_slope", "inplace"],
    doc: nnDoc("LeakyReLU")
  },
  sigmoidNode: {
    className: "nn.Sigmoid",
    params: [],
    doc: nnDoc("Sigmoid")
  },
  tanhNode: {
    className: "nn.Tanh",
    params: [],
    doc: nnDoc("Tanh")
  },
  eluNode: {
    className: "nn.ELU",
    params: ["alpha", "inplace"],
    doc: nnDoc("ELU")
  },
  seluNode: {
    className: "nn.SELU",
    params: ["inplace"],
    doc: nnDoc("SELU")
  },
  geluNode: {
    className: "nn.GELU",
    params: ["approximate"],
    doc: nnDoc("GELU")
  },
  siluNode: {
    className: "nn.SiLU",
    params: ["inplace"],
    doc: nnDoc("SiLU")
  },
  mishNode: {
    className: "nn.Mish",
    params: ["inplace"],
    doc: nnDoc("Mish")
  },
  hardswishNode: {
    className: "nn.Hardswish",
    params: ["inplace"],
    doc: nnDoc("Hardswish")
  },
  hardsigmoidNode: {
    className: "nn.Hardsigmoid",
    params: ["inplace"],
    doc: nnDoc("Hardsigmoid")
  },
  softmaxNode: {
    className: "nn.Softmax",
    params: ["dim"],
    doc: nnDoc("Softmax")
  },
  logsoftmaxNode: {
    className: "nn.LogSoftmax",
    params: ["dim"],
    doc: nnDoc("LogSoftmax")
  },
  dropoutNode: {
    className: "nn.Dropout",
    params: ["p", "inplace"],
    doc: nnDoc("Dropout")
  },
  dropout2dNode: {
    className: "nn.Dropout2d",
    params: ["p", "inplace"],
    doc: nnDoc("Dropout2d")
  },
  dropout3dNode: {
    className: "nn.Dropout3d",
    params: ["p", "inplace"],
    doc: nnDoc("Dropout3d")
  },
  lstmNode: {
    className: "nn.LSTM",
    params: ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional"],
    doc: nnDoc("LSTM")
  },
  gruNode: {
    className: "nn.GRU",
    params: ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional"],
    doc: nnDoc("GRU")
  },
  rnnNode: {
    className: "nn.RNN",
    params: ["input_size", "hidden_size", "num_layers", "nonlinearity", "bias", "batch_first", "dropout", "bidirectional"],
    doc: nnDoc("RNN")
  },
  multiheadattentionNode: {
    className: "nn.MultiheadAttention",
    params: ["embed_dim", "num_heads", "dropout", "bias", "add_bias_kv", "add_zero_attn", "kdim", "vdim"],
    doc: nnDoc("MultiheadAttention")
  },
  transformerencoderlayerNode: {
    className: "nn.TransformerEncoderLayer",
    params: ["d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps", "batch_first", "norm_first"],
    doc: nnDoc("TransformerEncoderLayer")
  },
  transformerdecoderlayerNode: {
    className: "nn.TransformerDecoderLayer",
    params: ["d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps", "batch_first", "norm_first"],
    doc: nnDoc("TransformerDecoderLayer")
  },
  flattenNode: {
    className: "nn.Flatten",
    params: ["start_dim", "end_dim"],
    doc: nnDoc("Flatten")
  },
  upsampleNode: {
    className: "nn.Upsample",
    params: ["size", "scale_factor", "mode", "align_corners"],
    doc: nnDoc("Upsample")
  },
  convtranspose1dNode: {
    className: "nn.ConvTranspose1d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"],
    doc: nnDoc("ConvTranspose1d")
  },
  convtranspose2dNode: {
    className: "nn.ConvTranspose2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"],
    doc: nnDoc("ConvTranspose2d")
  },
  convtranspose3dNode: {
    className: "nn.ConvTranspose3d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"],
    doc: nnDoc("ConvTranspose3d")
  },
  depthwiseconv2dNode: {
    className: "nn.Conv2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "bias"],
    doc: nnDoc("Conv2d")
  },
  separableconv2dNode: {
    className: null, // Special handling in generator (depthwise + pointwise Conv2d)
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    doc: nnDoc("Conv2d")
  },
  mbconvNode: {
    className: null, // Special handling in generator
    params: ["in_channels", "out_channels", "kernel_size", "stride", "expand_ratio", "se_ratio"],
    doc: null
  },
  moeNode: {
    // Special handling in generator: emits a reusable MixtureOfExperts helper class.
    // Sparse Mixture-of-Experts feed-forward block (Mixtral / DeepSeek style).
    className: null,
    params: ["d_model", "d_ff", "num_experts", "top_k", "activation"],
    doc: "https://docs.pytorch.org/docs/stable/generated/torch.topk.html"
  },
  addNode: {
    className: null, // Special handling in generator
    params: [],
    doc: null
  },
  multiplyNode: {
    className: null, // Special handling in generator
    params: [],
    doc: null
  },
  concatenateNode: {
    className: null, // Special handling in generator (torch.cat)
    params: ["dim"],
    doc: `${PYTORCH_DOCS_BASE}/generated/torch.cat.html`
  }
}
