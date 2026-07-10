import type { TensorShape } from "./tensor-shape-calculator"

export interface LayerAnalysis {
  name: string
  type: string
  parameters: number
  flops: number
  memoryMB: number
  inputShape: TensorShape
  outputShape: TensorShape
}

export interface ModelAnalysis {
  totalParameters: number
  trainableParameters: number
  totalFLOPs: number
  memoryUsageMB: number
  layers: LayerAnalysis[]
  modelSizeMB: number
  estimatedInferenceTimeMs: number
}

// kernel_size may be a number, an array like [3, 3], or a string like "3,3".
// Normalize to per-axis sizes so parameter/FLOP math never produces NaN.
function parseKernelSizes(value: any, dims: number, defaultValue: number): number[] {
  let parts: number[] = []
  if (typeof value === "number" && !isNaN(value)) {
    parts = [value]
  } else if (Array.isArray(value)) {
    parts = value.map(Number).filter((n) => !isNaN(n))
  } else if (typeof value === "string") {
    parts = value
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n))
  }
  if (parts.length === 0) {
    parts = [defaultValue]
  }
  return Array.from({ length: dims }, (_, i) => parts[i] ?? parts[0])
}

export function analyzeLayer(
  nodeType: string,
  nodeData: any,
  inputShape: TensorShape,
  outputShape: TensorShape,
): LayerAnalysis {
  const analysis: LayerAnalysis = {
    name: nodeData.name || `${nodeType}_layer`,
    type: nodeType,
    parameters: 0,
    flops: 0,
    memoryMB: 0,
    inputShape,
    outputShape,
  }

  const batchSize = 1

  switch (nodeType) {
    case "conv2dNode":
    case "depthwiseconv2dNode":
      const inChannels = nodeData.in_channels || 3
      const outChannels = nodeData.out_channels || 32
      const [kernelH, kernelW] = parseKernelSizes(nodeData.kernel_size, 2, 3)
      const groups = nodeData.groups || 1

      // Parameters: weights + bias
      analysis.parameters = (inChannels * outChannels * kernelH * kernelW) / groups + outChannels

      // FLOPs: for each output pixel, kernel_h * kernel_w * in_channels * out_channels operations
      if (typeof outputShape.height === "number" && typeof outputShape.width === "number") {
        analysis.flops =
          (batchSize * outputShape.height * outputShape.width * kernelH * kernelW * inChannels * outChannels) /
          groups
      }

      // Memory: input + output + weights
      const inputMemory =
        batchSize *
        inChannels *
        (typeof inputShape.height === "number" ? inputShape.height : 1) *
        (typeof inputShape.width === "number" ? inputShape.width : 1) *
        4 // 4 bytes per float32
      const outputMemory =
        batchSize *
        outChannels *
        (typeof outputShape.height === "number" ? outputShape.height : 1) *
        (typeof outputShape.width === "number" ? outputShape.width : 1) *
        4
      const weightMemory = analysis.parameters * 4
      analysis.memoryMB = (inputMemory + outputMemory + weightMemory) / (1024 * 1024)
      break

    case "conv1dNode":
      const conv1dIn = nodeData.in_channels || 1
      const conv1dOut = nodeData.out_channels || 32
      const [conv1dKernel] = parseKernelSizes(nodeData.kernel_size, 1, 3)

      analysis.parameters = conv1dIn * conv1dOut * conv1dKernel + conv1dOut

      if (typeof outputShape.length === "number") {
        analysis.flops = batchSize * outputShape.length * conv1dKernel * conv1dIn * conv1dOut
      }
      break

    case "conv3dNode":
      const conv3dIn = nodeData.in_channels || 3
      const conv3dOut = nodeData.out_channels || 32
      const [conv3dKd, conv3dKh, conv3dKw] = parseKernelSizes(nodeData.kernel_size, 3, 3)

      analysis.parameters = conv3dIn * conv3dOut * conv3dKd * conv3dKh * conv3dKw + conv3dOut

      if (
        typeof outputShape.depth === "number" &&
        typeof outputShape.height === "number" &&
        typeof outputShape.width === "number"
      ) {
        analysis.flops =
          batchSize *
          outputShape.depth *
          outputShape.height *
          outputShape.width *
          conv3dKd *
          conv3dKh *
          conv3dKw *
          conv3dIn *
          conv3dOut
      }
      break

    case "linearNode":
      const inFeatures = nodeData.in_features || 128
      const outFeatures = nodeData.out_features || 64

      analysis.parameters = inFeatures * outFeatures + outFeatures // weights + bias
      analysis.flops = batchSize * inFeatures * outFeatures * 2 // multiply-add operations

      const linearInputMem = batchSize * inFeatures * 4
      const linearOutputMem = batchSize * outFeatures * 4
      const linearWeightMem = analysis.parameters * 4
      analysis.memoryMB = (linearInputMem + linearOutputMem + linearWeightMem) / (1024 * 1024)
      break

    case "multiheadattentionNode":
      const embedDim = nodeData.embed_dim || 128
      const numHeads = nodeData.num_heads || 8
      const seqLen = typeof inputShape.sequence === "number" ? inputShape.sequence : 100 // assume sequence length

      // Parameters: Q, K, V projections + output projection
      analysis.parameters = 4 * embedDim * embedDim + 4 * embedDim // 4 linear layers

      // FLOPs: attention computation is O(seq_len^2 * embed_dim)
      analysis.flops = batchSize * numHeads * seqLen * seqLen * embedDim + batchSize * seqLen * embedDim * embedDim * 4

      const attentionMem = batchSize * numHeads * seqLen * seqLen * 4 // attention weights
      const projectionMem = batchSize * seqLen * embedDim * 4 * 4 // Q, K, V, output
      analysis.memoryMB = (attentionMem + projectionMem) / (1024 * 1024)
      break

    case "batchnorm2dNode":
    case "batchnorm1dNode":
      const bnFeatures = nodeData.num_features || 32
      analysis.parameters = bnFeatures * 4 // gamma, beta, running_mean, running_var

      if (typeof outputShape.height === "number" && typeof outputShape.width === "number") {
        analysis.flops =
          batchSize *
          bnFeatures *
          (typeof outputShape.height === "number" ? outputShape.height : 1) *
          (typeof outputShape.width === "number" ? outputShape.width : 1) *
          2 // normalize + scale
      }
      break

    case "layernormNode":
      const lnShape = nodeData.normalized_shape || [128]
      const lnFeatures = Array.isArray(lnShape) ? lnShape.reduce((a, b) => a * b, 1) : lnShape
      analysis.parameters = lnFeatures * 2 // gamma, beta
      analysis.flops = batchSize * lnFeatures * 5 // mean, var, normalize, scale, shift
      break

    case "rmsnormNode":
      // RMSNorm has only a learnable scale (weight) when elementwise_affine=True,
      // and no bias/mean subtraction (that is what distinguishes it from LayerNorm).
      const rmsShape = nodeData.normalized_shape || [128]
      const rmsFeatures = Array.isArray(rmsShape) ? rmsShape.reduce((a, b) => a * b, 1) : rmsShape
      analysis.parameters = nodeData.elementwise_affine === false ? 0 : rmsFeatures // gamma only
      analysis.flops = batchSize * rmsFeatures * 3 // square-mean, rsqrt-normalize, scale
      break

    case "moeNode": {
      // Sparse MoE: all experts hold parameters, but only top_k run per token.
      const dModel = nodeData.d_model || 512
      const dFf = nodeData.d_ff || dModel * 4
      const numExperts = nodeData.num_experts || 8
      const topK = nodeData.top_k || 2
      const expertParams = 2 * dModel * dFf + dFf + dModel // two linears (+biases)
      const gateParams = dModel * numExperts // router (no bias)
      analysis.parameters = numExperts * expertParams + gateParams
      // Only the selected top_k experts execute per token -> active FLOPs.
      const seqLen = typeof inputShape.sequence === "number" ? inputShape.sequence : 1
      const activeParams = topK * expertParams + gateParams
      analysis.flops = batchSize * seqLen * activeParams * 2
      break
    }

    case "lstmNode":
    case "gruNode":
    case "rnnNode":
      const inputSize = nodeData.input_size || 128
      const hiddenSize = nodeData.hidden_size || 64
      const numLayers = nodeData.num_layers || 1

      if (nodeType === "lstmNode") {
        // LSTM has 4 gates, each with input-to-hidden and hidden-to-hidden weights
        analysis.parameters = numLayers * (4 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * 2))
      } else if (nodeType === "gruNode") {
        // GRU has 3 gates
        analysis.parameters = numLayers * (3 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * 2))
      } else { // rnnNode
        // Simple RNN has 1 "gate"
        analysis.parameters = numLayers * (1 * (inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize * 2))
      }

      const rnnSeqLen = typeof inputShape.sequence === "number" ? inputShape.sequence : 100
      analysis.flops = batchSize * rnnSeqLen * analysis.parameters * 2 // approximate
      break

    // Activation functions and pooling have no parameters
    case "reluNode":
    case "sigmoidNode":
    case "tanhNode":
    case "geluNode":
    case "siluNode":
    case "mishNode":
    case "hardswishNode":
    case "hardsigmoidNode":
    case "softmaxNode":
    case "leakyreluNode":
    case "maxpool2dNode":
    case "avgpool2dNode":
    case "adaptiveavgpool2dNode":
    case "dropoutNode":
    case "addNode":
    case "multiplyNode":
      analysis.parameters = 0
      // Minimal FLOPs for activation functions
      if (
        typeof outputShape.height === "number" &&
        typeof outputShape.width === "number" &&
        typeof outputShape.channels === "number"
      ) {
        analysis.flops = batchSize * outputShape.channels * outputShape.height * outputShape.width
      }
      break

    default:
      analysis.parameters = 0
      analysis.flops = 0
  }

  return analysis
}

export function analyzeModel(nodes: any[], edges: any[]): ModelAnalysis {
  const layerAnalyses: LayerAnalysis[] = []
  let totalParameters = 0
  let totalFLOPs = 0
  let totalMemoryMB = 0

  // Analyze each layer
  for (const node of nodes) {
    if (node.type === "inputNode") continue

    const inputShape = node.data.inputShape || {}
    const outputShape = node.data.outputShape || inputShape

    const layerAnalysis = analyzeLayer(node.type, node.data, inputShape, outputShape)
    layerAnalyses.push(layerAnalysis)

    totalParameters += layerAnalysis.parameters
    totalFLOPs += layerAnalysis.flops
    totalMemoryMB += layerAnalysis.memoryMB
  }

  // Estimate model size (parameters * 4 bytes for float32)
  const modelSizeMB = (totalParameters * 4) / (1024 * 1024)

  // Rough inference time estimation (very approximate)
  // Based on typical GPU/CPU performance: ~1 TFLOP/s for CPU, ~10 TFLOP/s for GPU
  const estimatedInferenceTimeMs = (totalFLOPs / 1e12) * 1000 // Assume 1 TFLOP/s

  return {
    totalParameters,
    trainableParameters: totalParameters, // Assume all parameters are trainable
    totalFLOPs,
    memoryUsageMB: totalMemoryMB,
    layers: layerAnalyses,
    modelSizeMB,
    estimatedInferenceTimeMs,
  }
}

export function formatNumber(num: number): string {
  if (num >= 1e9) {
    return `${(num / 1e9).toFixed(2)}B`
  } else if (num >= 1e6) {
    return `${(num / 1e6).toFixed(2)}M`
  } else if (num >= 1e3) {
    return `${(num / 1e3).toFixed(2)}K`
  } else {
    return num.toFixed(0)
  }
}

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  } else if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  } else if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`
  } else {
    return `${bytes.toFixed(0)} B`
  }
}
