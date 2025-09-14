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

// LSTM/GRU data
export interface RNNNodeData extends BaseNodeData {
  input_size: number
  hidden_size: number
  num_layers?: number
  bias?: boolean
  batch_first?: boolean
  dropout?: number
  bidirectional?: boolean
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

// PyTorch layer manifest - defines layer parameters and class names
export const PYTORCH_LAYER_MANIFEST = {
  linearNode: {
    className: "nn.Linear",
    params: ["in_features", "out_features", "bias"]
  },
  conv1dNode: {
    className: "nn.Conv1d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"]
  },
  conv2dNode: {
    className: "nn.Conv2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"]
  },
  conv3dNode: {
    className: "nn.Conv3d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"]
  },
  maxpool1dNode: {
    className: "nn.MaxPool1d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
  },
  maxpool2dNode: {
    className: "nn.MaxPool2d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
  },
  maxpool3dNode: {
    className: "nn.MaxPool3d",
    params: ["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"]
  },
  avgpool1dNode: {
    className: "nn.AvgPool1d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"]
  },
  avgpool2dNode: {
    className: "nn.AvgPool2d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"]
  },
  avgpool3dNode: {
    className: "nn.AvgPool3d",
    params: ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"]
  },
  batchnorm1dNode: {
    className: "nn.BatchNorm1d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  batchnorm2dNode: {
    className: "nn.BatchNorm2d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  batchnorm3dNode: {
    className: "nn.BatchNorm3d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  layernormNode: {
    className: "nn.LayerNorm",
    params: ["normalized_shape", "eps", "elementwise_affine"]
  },
  instancenorm1dNode: {
    className: "nn.InstanceNorm1d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  instancenorm2dNode: {
    className: "nn.InstanceNorm2d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  instancenorm3dNode: {
    className: "nn.InstanceNorm3d",
    params: ["num_features", "eps", "momentum", "affine", "track_running_stats"]
  },
  groupnormNode: {
    className: "nn.GroupNorm",
    params: ["num_groups", "num_channels", "eps", "affine"]
  },
  reluNode: {
    className: "nn.ReLU",
    params: ["inplace"]
  },
  leakyreluNode: {
    className: "nn.LeakyReLU",
    params: ["negative_slope", "inplace"]
  },
  sigmoidNode: {
    className: "nn.Sigmoid",
    params: []
  },
  tanhNode: {
    className: "nn.Tanh",
    params: []
  },
  eluNode: {
    className: "nn.ELU",
    params: ["alpha", "inplace"]
  },
  seluNode: {
    className: "nn.SELU",
    params: ["inplace"]
  },
  geluNode: {
    className: "nn.GELU",
    params: []
  },
  softmaxNode: {
    className: "nn.Softmax",
    params: ["dim"]
  },
  dropoutNode: {
    className: "nn.Dropout",
    params: ["p", "inplace"]
  },
  dropout2dNode: {
    className: "nn.Dropout2d",
    params: ["p", "inplace"]
  },
  dropout3dNode: {
    className: "nn.Dropout3d",
    params: ["p", "inplace"]
  },
  lstmNode: {
    className: "nn.LSTM",
    params: ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional"]
  },
  gruNode: {
    className: "nn.GRU",
    params: ["input_size", "hidden_size", "num_layers", "bias", "batch_first", "dropout", "bidirectional"]
  },
  rnnNode: {
    className: "nn.RNN",
    params: ["input_size", "hidden_size", "num_layers", "nonlinearity", "bias", "batch_first", "dropout", "bidirectional"]
  },
  multiheadattentionNode: {
    className: "nn.MultiheadAttention",
    params: ["embed_dim", "num_heads", "dropout", "bias", "add_bias_kv", "add_zero_attn", "kdim", "vdim"]
  },
  transformerencoderlayerNode: {
    className: "nn.TransformerEncoderLayer",
    params: ["d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps", "batch_first", "norm_first"]
  },
  transformerdecoderlayerNode: {
    className: "nn.TransformerDecoderLayer",
    params: ["d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps", "batch_first", "norm_first"]
  },
  flattenNode: {
    className: "nn.Flatten",
    params: ["start_dim", "end_dim"]
  },
  upsampleNode: {
    className: "nn.Upsample",
    params: ["size", "scale_factor", "mode", "align_corners"]
  },
  convtranspose1dNode: {
    className: "nn.ConvTranspose1d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"]
  },
  convtranspose2dNode: {
    className: "nn.ConvTranspose2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"]
  },
  convtranspose3dNode: {
    className: "nn.ConvTranspose3d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation"]
  },
  depthwiseconv2dNode: {
    className: "nn.Conv2d",
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "bias"]
  },
  separableconv2dNode: {
    className: null, // Special handling in generator
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
  },
  addNode: {
    className: null, // Special handling in generator
    params: []
  },
  concatenateNode: {
    className: null, // Special handling in generator
    params: ["dim"]
  }
}
