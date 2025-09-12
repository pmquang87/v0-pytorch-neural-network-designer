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

export interface ShapeValidationResult {
  isValid: boolean
  error?: string
  warning?: string
}

export function validateTensorShapes(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): ShapeValidationResult {
  switch (nodeType) {
    case "addNode":
      if (inputShapes.length !== 2) {
        return { isValid: false, error: "Add operation requires exactly 2 inputs" }
      }
      const [shape1, shape2] = inputShapes
      if (!shapesCompatible(shape1, shape2)) {
        return { isValid: false, error: "Input shapes must be identical for element-wise addition" }
      }
      break

    case "concatenateNode":
      if (inputShapes.length < 2) {
        return { isValid: false, error: "Concatenate operation requires at least 2 inputs" }
      }
      const dim = nodeData.dim || 1
      if (!canConcatenate(inputShapes, dim)) {
        return { isValid: false, error: `Cannot concatenate tensors along dimension ${dim}` }
      }
      break

    case "linearNode":
      if (inputShapes[0]?.features === undefined && inputShapes[0]?.channels === undefined) {
        return { isValid: false, error: "Linear layer requires flattened input or feature dimension" }
      }
      break

    case "multiheadattentionNode":
      const embedDim = nodeData.embed_dim || 128
      if (inputShapes[0]?.features !== embedDim && inputShapes[0]?.channels !== embedDim) {
        return {
          isValid: false,
          error: `MultiheadAttention expects input dimension ${embedDim}, got ${inputShapes[0]?.features || inputShapes[0]?.channels}`,
        }
      }
      break
  }

  return { isValid: true }
}

function shapesCompatible(shape1: TensorShape, shape2: TensorShape): boolean {
  const keys = new Set([...Object.keys(shape1), ...Object.keys(shape2)])
  for (const key of keys) {
    const val1 = (shape1 as any)[key]
    const val2 = (shape2 as any)[key]
    if (val1 !== undefined && val2 !== undefined && val1 !== val2 && val1 !== "dynamic" && val2 !== "dynamic") {
      return false
    }
  }
  return true
}

function canConcatenate(shapes: TensorShape[], dim: number): boolean {
  if (shapes.length < 2) return false

  const dimNames = ["batch", "channels", "height", "width", "depth", "features", "sequence", "length"]
  const targetDim = dimNames[dim]

  for (let i = 1; i < shapes.length; i++) {
    for (const key of Object.keys(shapes[0])) {
      if (key !== targetDim) {
        const val1 = (shapes[0] as any)[key]
        const val2 = (shapes[i] as any)[key]
        if (val1 !== val2 && val1 !== "dynamic" && val2 !== "dynamic") {
          return false
        }
      }
    }
  }
  return true
}

export function calculateOutputShape(nodeType: string, inputShape: TensorShape, nodeData: any): TensorShape {
  switch (nodeType) {
    case "inputNode":
      return {
        batch: nodeData.batch_size ?? 1,
        channels: nodeData.channels ?? 3,
        height: nodeData.height ?? 28,
        width: nodeData.width ?? 28,
      }

    case "conv2dNode":
    case "depthwiseconv2dNode":
      const padding = nodeData.padding || 0
      const stride = nodeData.stride || 1
      const kernel = nodeData.kernel_size || 3

      if (inputShape.height && inputShape.width) {
        const newHeight = Math.floor((Number(inputShape.height) + 2 * padding - kernel) / stride + 1)
        const newWidth = Math.floor((Number(inputShape.width) + 2 * padding - kernel) / stride + 1)
        return {
          batch: inputShape.batch,
          channels: nodeData.out_channels || 32,
          height: Math.max(1, newHeight),
          width: Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "conv1dNode":
      const conv1dPadding = nodeData.padding || 0
      const conv1dStride = nodeData.stride || 1
      const conv1dKernel = nodeData.kernel_size || 3

      if (inputShape.length) {
        const newLength = Math.floor((Number(inputShape.length) + 2 * conv1dPadding - conv1dKernel) / conv1dStride + 1)
        return {
          batch: inputShape.batch,
          channels: nodeData.out_channels || 32,
          length: Math.max(1, newLength),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "conv3dNode":
      const conv3dPadding = nodeData.padding || 0
      const conv3dStride = nodeData.stride || 1
      const conv3dKernel = nodeData.kernel_size || 3

      if (inputShape.depth && inputShape.height && inputShape.width) {
        const newDepth = Math.floor((Number(inputShape.depth) + 2 * conv3dPadding - conv3dKernel) / conv3dStride + 1)
        const newHeight = Math.floor((Number(inputShape.height) + 2 * conv3dPadding - conv3dKernel) / conv3dStride + 1)
        const newWidth = Math.floor((Number(inputShape.width) + 2 * conv3dPadding - conv3dKernel) / conv3dStride + 1)
        return {
          batch: inputShape.batch,
          channels: nodeData.out_channels || 32,
          depth: Math.max(1, newDepth),
          height: Math.max(1, newHeight),
          width: Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "convtranspose2dNode":
      const transStride = nodeData.stride || 2
      const transKernel = nodeData.kernel_size || 2
      const transPadding = nodeData.padding || 0
      const outputPadding = nodeData.output_padding || 0

      if (inputShape.height && inputShape.width) {
        const newHeight = (Number(inputShape.height) - 1) * transStride - 2 * transPadding + transKernel + outputPadding
        const newWidth = (Number(inputShape.width) - 1) * transStride - 2 * transPadding + transKernel + outputPadding
        return {
          batch: inputShape.batch,
          channels: nodeData.out_channels || 16,
          height: Math.max(1, newHeight),
          width: Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 16 }

    case "linearNode":
      return {
        batch: inputShape.batch,
        features: nodeData.out_features || 64,
      }

    case "maxpool2dNode":
    case "avgpool2dNode":
      const poolKernel = nodeData.kernel_size || 2
      const poolStride = nodeData.stride || poolKernel

      if (inputShape.height && inputShape.width) {
        return {
          batch: inputShape.batch,
          channels: inputShape.channels,
          height: Math.floor(Number(inputShape.height) / poolStride),
          width: Math.floor(Number(inputShape.width) / poolStride),
        }
      }
      return inputShape

    case "maxpool1dNode":
    case "avgpool1dNode":
      const pool1dKernel = nodeData.kernel_size || 2
      const pool1dStride = nodeData.stride || pool1dKernel

      if (inputShape.length) {
        return {
          batch: inputShape.batch,
          channels: inputShape.channels,
          length: Math.floor(Number(inputShape.length) / pool1dStride),
        }
      }
      return inputShape

    case "maxpool3dNode":
    case "avgpool3dNode":
      const pool3dKernel = nodeData.kernel_size || 2
      const pool3dStride = nodeData.stride || pool3dKernel

      if (inputShape.depth && inputShape.height && inputShape.width) {
        return {
          batch: inputShape.batch,
          channels: inputShape.channels,
          depth: Math.floor(Number(inputShape.depth) / pool3dStride),
          height: Math.floor(Number(inputShape.height) / pool3dStride),
          width: Math.floor(Number(inputShape.width) / pool3dStride),
        }
      }
      return inputShape

    case "adaptivemaxpool1dNode":
      return {
        batch: inputShape.batch,
        channels: inputShape.channels,
        length: nodeData.output_size || 1,
      }

    case "adaptivemaxpool3dNode":
      const output3d = nodeData.output_size || [1, 1, 1]
      return {
        batch: inputShape.batch,
        channels: inputShape.channels,
        depth: output3d[0],
        height: output3d[1],
        width: output3d[2],
      }

    case "fractionalmaxpool2dNode":
      const outputRatio = nodeData.output_ratio || 0.5
      if (inputShape.height && inputShape.width) {
        return {
          batch: inputShape.batch,
          channels: inputShape.channels,
          height: Math.floor(Number(inputShape.height) * outputRatio),
          width: Math.floor(Number(inputShape.width) * outputRatio),
        }
      }
      return inputShape

    case "lppool2dNode":
      const lpKernel = nodeData.kernel_size || 2
      const lpStride = nodeData.stride || lpKernel

      if (inputShape.height && inputShape.width) {
        return {
          batch: inputShape.batch,
          channels: inputShape.channels,
          height: Math.floor(Number(inputShape.height) / lpStride),
          width: Math.floor(Number(inputShape.width) / lpStride),
        }
      }
      return inputShape

    case "flattenNode":
      const startDim = nodeData.start_dim || 1
      const endDim = nodeData.end_dim || -1

      // Calculate total features from flattened dimensions
      let totalFeatures = 1
      if (inputShape.channels) totalFeatures *= Number(inputShape.channels)
      if (inputShape.height) totalFeatures *= Number(inputShape.height)
      if (inputShape.width) totalFeatures *= Number(inputShape.width)
      if (inputShape.depth) totalFeatures *= Number(inputShape.depth)

      return {
        batch: inputShape.batch,
        features: totalFeatures,
      }

    case "adaptiveavgpool2dNode":
      const outputSize = nodeData.output_size || [1, 1]
      return {
        batch: inputShape.batch,
        channels: inputShape.channels,
        height: outputSize[0],
        width: outputSize[1],
      }

    case "multiheadattentionNode":
      const embedDim = nodeData.embed_dim || 128
      return {
        batch: inputShape.batch,
        sequence: inputShape.sequence,
        features: embedDim,
      }

    case "transformerencoderlayerNode":
    case "transformerdecoderlayerNode":
      return inputShape // Transformer layers preserve input shape

    case "layernormNode":
      return inputShape

    case "groupnormNode":
      return inputShape

    case "instancenorm1dNode":
    case "instancenorm2dNode":
    case "instancenorm3dNode":
      return inputShape

    // Activation functions preserve shape
    case "reluNode":
    case "sigmoidNode":
    case "tanhNode":
    case "softmaxNode":
    case "leakyreluNode":
    case "geluNode":
    case "siluNode":
    case "mishNode":
    case "hardswishNode":
    case "hardsigmoidNode":
    case "dropoutNode":
      return inputShape

    // Normalization layers preserve shape
    case "batchnorm1dNode":
    case "batchnorm2dNode":
      return inputShape

    case "addNode":
      return inputShape // Element-wise addition preserves shape

    case "concatenateNode":
      const dim = nodeData.dim || 1
      if (dim === 1 && inputShape.channels) {
        return { ...inputShape, channels: Number(inputShape.channels) * 2 } // Simplified assumption for 2 inputs
      }
      return inputShape

    case "separableconv2dNode":
      const sepPadding = nodeData.padding || 0
      const sepStride = nodeData.stride || 1
      const sepKernel = nodeData.kernel_size || 3

      if (inputShape.height && inputShape.width) {
        const newHeight = Math.floor((Number(inputShape.height) + 2 * sepPadding - sepKernel) / sepStride + 1)
        const newWidth = Math.floor((Number(inputShape.width) + 2 * sepPadding - sepKernel) / sepStride + 1)
        return {
          batch: inputShape.batch,
          channels: nodeData.out_channels || 64,
          height: Math.max(1, newHeight),
          width: Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 64 }

    default:
      return {
        batch: inputShape.batch ?? 1,
        channels: inputShape.channels ?? "dynamic",
        height: inputShape.height ?? "dynamic",
        width: inputShape.width ?? "dynamic",
        depth: inputShape.depth ?? "dynamic",
        features: inputShape.features ?? "dynamic",
        sequence: inputShape.sequence ?? "dynamic",
        length: inputShape.length ?? "dynamic",
      }
  }
}

export function formatTensorShape(shape: TensorShape | {} | undefined): string {
  // Handle empty objects or undefined shapes
  if (!shape || Object.keys(shape).length === 0) {
    return "[?]"
  }

  const typedShape = shape as TensorShape
  const parts: string[] = []

  // Prioritize 2D convolutional dimensions if available
  if (typedShape.batch !== undefined) parts.push(`${typedShape.batch}`)
  if (typedShape.channels !== undefined) parts.push(`${typedShape.channels}`)
  if (typedShape.height !== undefined) parts.push(`${typedShape.height}`)
  if (typedShape.width !== undefined) parts.push(`${typedShape.width}`)

  // Add other dimensions only if they are the primary ones or if 2D dims are not present
  if (
    parts.length === 0 ||
    (typedShape.features !== undefined &&
      typedShape.channels === undefined &&
      typedShape.height === undefined &&
      typedShape.width === undefined)
  ) {
    if (typedShape.depth !== undefined) parts.push(`${typedShape.depth}`)
    if (typedShape.features !== undefined) parts.push(`${typedShape.features}`)
    if (typedShape.sequence !== undefined) parts.push(`${typedShape.sequence}`)
    if (typedShape.length !== undefined) parts.push(`${typedShape.length}`)
  }

  // If no valid parts found, return placeholder
  if (parts.length === 0) {
    return "[?]"
  }

  return `[${parts.join(", ")}]`
}

export function inferDynamicShape(nodeType: string, inputShapes: TensorShape[], nodeData: any): TensorShape {
  // For operations that can work with dynamic shapes
  if (nodeType === "addNode" || nodeType === "concatenateNode") {
    return calculateOutputShape(nodeType, inputShapes[0], nodeData)
  }

  // For most operations, use the first input shape
  return calculateOutputShape(nodeType, inputShapes[0] || { batch: "dynamic" }, nodeData)
}
