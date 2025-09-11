// Types for the JSON Intermediate Representation (IR)
export interface NodeData {
  [key: string]: any
}

export interface GraphNode {
  id: string
  type: string
  data: NodeData
}

export interface GraphEdge {
  id: string
  source: string
  target: string
  sourceHandle?: string
  targetHandle?: string
}

export interface GraphIR {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface GenerateModelRequest {
  graph: GraphIR
}

export interface GenerateModelResponse {
  success: boolean
  code?: string
  error?: string
}

// PyTorch layer manifest - maps node types to PyTorch classes
export const PYTORCH_LAYER_MANIFEST = {
  inputNode: {
    className: null, // Special case - not a PyTorch layer
    imports: [],
  },
  linearNode: {
    className: "nn.Linear",
    imports: ["torch.nn as nn"],
    params: ["in_features", "out_features", "bias"],
  },
  conv1dNode: {
    className: "nn.Conv1d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  conv2dNode: {
    className: "nn.Conv2d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  conv3dNode: {
    className: "nn.Conv3d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  convtranspose1dNode: {
    className: "nn.ConvTranspose1d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  convtranspose2dNode: {
    className: "nn.ConvTranspose2d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  convtranspose3dNode: {
    className: "nn.ConvTranspose3d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
  reluNode: {
    className: "nn.ReLU",
    imports: ["torch.nn as nn"],
    params: [],
  },
  leakyreluNode: {
    className: "nn.LeakyReLU",
    imports: ["torch.nn as nn"],
    params: ["negative_slope"],
  },
  sigmoidNode: {
    className: "nn.Sigmoid",
    imports: ["torch.nn as nn"],
    params: [],
  },
  tanhNode: {
    className: "nn.Tanh",
    imports: ["torch.nn as nn"],
    params: [],
  },
  geluNode: {
    className: "nn.GELU",
    imports: ["torch.nn as nn"],
    params: [],
  },
  siluNode: {
    className: "nn.SiLU",
    imports: ["torch.nn as nn"],
    params: [],
  },
  mishNode: {
    className: "nn.Mish",
    imports: ["torch.nn as nn"],
    params: [],
  },
  hardswishNode: {
    className: "nn.Hardswish",
    imports: ["torch.nn as nn"],
    params: [],
  },
  hardsigmoidNode: {
    className: "nn.Hardsigmoid",
    imports: ["torch.nn as nn"],
    params: [],
  },
  softmaxNode: {
    className: "nn.Softmax",
    imports: ["torch.nn as nn"],
    params: ["dim"],
  },
  maxpool2dNode: {
    className: "nn.MaxPool2d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  avgpool2dNode: {
    className: "nn.AvgPool2d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  adaptiveavgpool2dNode: {
    className: "nn.AdaptiveAvgPool2d",
    imports: ["torch.nn as nn"],
    params: ["output_size"],
  },
  adaptivemaxpool1dNode: {
    className: "nn.AdaptiveMaxPool1d",
    imports: ["torch.nn as nn"],
    params: ["output_size"],
  },
  adaptivemaxpool3dNode: {
    className: "nn.AdaptiveMaxPool3d",
    imports: ["torch.nn as nn"],
    params: ["output_size"],
  },
  fractionalmaxpool2dNode: {
    className: "nn.FractionalMaxPool2d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "output_ratio"],
  },
  fractionalmaxpool3dNode: {
    className: "nn.FractionalMaxPool3d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "output_ratio"],
  },
  lppool1dNode: {
    className: "nn.LPPool1d",
    imports: ["torch.nn as nn"],
    params: ["norm_type", "kernel_size", "stride"],
  },
  lppool2dNode: {
    className: "nn.LPPool2d",
    imports: ["torch.nn as nn"],
    params: ["norm_type", "kernel_size", "stride"],
  },
  lppool3dNode: {
    className: "nn.LPPool3d",
    imports: ["torch.nn as nn"],
    params: ["norm_type", "kernel_size", "stride"],
  },
  maxpool1dNode: {
    className: "nn.MaxPool1d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  maxpool3dNode: {
    className: "nn.MaxPool3d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  avgpool1dNode: {
    className: "nn.AvgPool1d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  avgpool3dNode: {
    className: "nn.AvgPool3d",
    imports: ["torch.nn as nn"],
    params: ["kernel_size", "stride"],
  },
  batchnorm1dNode: {
    className: "nn.BatchNorm1d",
    imports: ["torch.nn as nn"],
    params: ["num_features"],
  },
  batchnorm2dNode: {
    className: "nn.BatchNorm2d",
    imports: ["torch.nn as nn"],
    params: ["num_features"],
  },
  layernormNode: {
    className: "nn.LayerNorm",
    imports: ["torch.nn as nn"],
    params: ["normalized_shape"],
  },
  groupnormNode: {
    className: "nn.GroupNorm",
    imports: ["torch.nn as nn"],
    params: ["num_groups", "num_channels"],
  },
  instancenorm1dNode: {
    className: "nn.InstanceNorm1d",
    imports: ["torch.nn as nn"],
    params: ["num_features"],
  },
  instancenorm2dNode: {
    className: "nn.InstanceNorm2d",
    imports: ["torch.nn as nn"],
    params: ["num_features"],
  },
  instancenorm3dNode: {
    className: "nn.InstanceNorm3d",
    imports: ["torch.nn as nn"],
    params: ["num_features"],
  },
  dropoutNode: {
    className: "nn.Dropout",
    imports: ["torch.nn as nn"],
    params: ["p"],
  },
  flattenNode: {
    className: "nn.Flatten",
    imports: ["torch.nn as nn"],
    params: ["start_dim", "end_dim"],
  },
  lstmNode: {
    className: "nn.LSTM",
    imports: ["torch.nn as nn"],
    params: ["input_size", "hidden_size", "num_layers"],
  },
  gruNode: {
    className: "nn.GRU",
    imports: ["torch.nn as nn"],
    params: ["input_size", "hidden_size", "num_layers"],
  },
  multiheadattentionNode: {
    className: "nn.MultiheadAttention",
    imports: ["torch.nn as nn"],
    params: ["embed_dim", "num_heads", "dropout", "batch_first"],
  },
  transformerencoderlayerNode: {
    className: "nn.TransformerEncoderLayer",
    imports: ["torch.nn as nn"],
    params: ["d_model", "nhead", "dim_feedforward", "dropout"],
  },
  transformerdecoderlayerNode: {
    className: "nn.TransformerDecoderLayer",
    imports: ["torch.nn as nn"],
    params: ["d_model", "nhead", "dim_feedforward", "dropout"],
  },
  addNode: {
    className: null, // Special case - handled in forward pass
    imports: [],
    params: [],
  },
  concatenateNode: {
    className: null, // Special case - handled in forward pass
    imports: [],
    params: ["dim"],
  },
  depthwiseconv2dNode: {
    className: "nn.Conv2d",
    imports: ["torch.nn as nn"],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "groups"],
  },
  separableconv2dNode: {
    className: null, // Special case - combination of depthwise + pointwise
    imports: [],
    params: ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
  },
} as const
