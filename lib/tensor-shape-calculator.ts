export interface TensorShape {
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

function parseTuple(value: any, defaultValue: number): [number, number] {
  if (value === undefined) return [defaultValue, defaultValue];
  if (typeof value === 'number') return [value, value];
  if (typeof value === 'string') {
    const parts = value.split(',').map(s => parseInt(s.trim(), 10));
    if (parts.length === 1 && !isNaN(parts[0])) return [parts[0], parts[0]];
    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) return [parts[0], parts[1]];
  }
  if (Array.isArray(value)) {
    if (value.length === 1 && !isNaN(Number(value[0]))) return [Number(value[0]), Number(value[0])];
    if (value.length === 2 && !isNaN(Number(value[0])) && !isNaN(Number(value[1]))) return [Number(value[0]), Number(value[1])];
  }
  return [defaultValue, defaultValue];
}

const CANONICAL_DIM_ORDER: (keyof TensorShape)[] = ["channels", "depth", "height", "width", "sequence", "length", "features"];

function getOrderedDimensions(shape: TensorShape): (keyof TensorShape)[] {
    return CANONICAL_DIM_ORDER.filter(dim => shape[dim] !== undefined);
}

export function validateTensorShapes(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): ShapeValidationResult {
  switch (nodeType) {
    case "addNode":
      if (inputShapes.length < 2) {
        return { isValid: true }; 
      }
      const [shape1, shape2] = inputShapes
      if (!shapesCompatible(shape1, shape2)) {
        return { isValid: false, error: "Addition error: Input shapes must be identical for element-wise addition." }
      }
      break

    case "concatenateNode":
      if (inputShapes.length < 2) {
        return { isValid: true }; // Not an error if not fully connected
      }
      const dim = nodeData.dim ?? 0;
      const firstShape = inputShapes[0];
      if (!firstShape) break;
      const orderedDims = getOrderedDimensions(firstShape);

      if (dim >= orderedDims.length) {
        return { isValid: false, error: `Invalid dimension ${dim} for a tensor with ${orderedDims.length} dimensions.` };
      }

      for (let i = 1; i < inputShapes.length; i++) {
        const currentShape = inputShapes[i];
        if (!currentShape) continue;
        const currentOrderedDims = getOrderedDimensions(currentShape);

        if (orderedDims.length !== currentOrderedDims.length) {
            return { isValid: false, error: `Concatenation error: All inputs must have the same number of dimensions. Input 1 has ${orderedDims.length}, but input ${i+1} has ${currentOrderedDims.length}.` };
        }

        for (let j = 0; j < orderedDims.length; j++) {
          if (j !== dim) {
            const dimName = orderedDims[j];
            const val1 = firstShape[dimName];
            const val2 = currentShape[dimName];
            if (val1 !== "dynamic" && val2 !== "dynamic" && val1 !== val2) {
              return { isValid: false, error: `Concatenation error: Input ${i+1} has a mismatch on dimension '${dimName}'. Expected size ${val1}, but got ${val2}. All non-concatenation dimensions must match.` };
            }
          }
        }
      }
      break

    case "linearNode":
      const in_features = nodeData.in_features
      const input_features = inputShapes[0]?.features || inputShapes[0]?.channels
      if (in_features && input_features && in_features !== input_features && input_features !== "dynamic") {
        return {
          isValid: false,
          error: `Input features ${input_features} do not match layer's in_features ${in_features}`,
        }
      }
      break

    case "multiheadattentionNode":
      const embedDim = nodeData.embed_dim || 128
      if (inputShapes[0]?.features !== embedDim && inputShapes[0]?.channels !== embedDim) {
        return {
          isValid: false,
          error: `MultiheadAttention expects input dimension ${embedDim}, got ${inputShapes[0]?.features ||
            inputShapes[0]?.channels}`,
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

export function calculateOutputShape(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): TensorShape {
  const inputShape = inputShapes[0] || {}
  switch (nodeType) {
    case "inputNode":
      return {
        channels: nodeData.channels,
        height: nodeData.height,
        width: nodeData.width ?? 28,
      }

    case "conv2dNode":
    case "depthwiseconv2dNode":
      const padding = parseTuple(nodeData.padding, 0)
      const stride = parseTuple(nodeData.stride, 1)
      const kernel = parseTuple(nodeData.kernel_size, 3)
      const dilation = parseTuple(nodeData.dilation, 1)

      if (inputShape.height && inputShape.width && stride[0] > 0 && stride[1] > 0) {
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1,
        )
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1,
        )
        return {
          channels: nodeData.out_channels || 32,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "conv1dNode":
      const conv1dPadding = nodeData.padding || 0
      const conv1dStride = nodeData.stride || 1
      const conv1dKernel = nodeData.kernel_size || 3
      const conv1dDilation = nodeData.dilation || 1

      if (inputShape.length) {
        const newLength = Math.floor(
          (Number(inputShape.length) + 2 * conv1dPadding - conv1dDilation * (conv1dKernel - 1) - 1) /
            conv1dStride +
            1,
        )
        return {
          channels: nodeData.out_channels || 32,
          length: isNaN(newLength) ? "dynamic" : Math.max(1, newLength),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "conv3dNode":
      const conv3dPadding = nodeData.padding || 0
      const conv3dStride = nodeData.stride || 1
      const conv3dKernel = nodeData.kernel_size || 3
      const conv3dDilation = nodeData.dilation || 1

      if (inputShape.depth && inputShape.height && inputShape.width) {
        const newDepth = Math.floor(
          (Number(inputShape.depth) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        )
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        )
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        )
        return {
          channels: nodeData.out_channels || 32,
          depth: isNaN(newDepth) ? "dynamic" : Math.max(1, newDepth),
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 }

    case "convtranspose2dNode":
      const transStride = nodeData.stride || 2
      const transKernel = nodeData.kernel_size || 2
      const transPadding = nodeData.padding || 0
      const outputPadding = nodeData.output_padding || 0
      const transDilation = nodeData.dilation || 1

      if (inputShape.height && inputShape.width) {
        const newHeight =
          (Number(inputShape.height) - 1) * transStride -
          2 * transPadding +
          transDilation * (transKernel - 1) +
          outputPadding +
          1
        const newWidth =
          (Number(inputShape.width) - 1) * transStride -
          2 * transPadding +
          transDilation * (transKernel - 1) +
          outputPadding +
          1
        return {
          channels: nodeData.out_channels || 16,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 16 }

    case "linearNode":
      return {
        features: nodeData.out_features || 64,
      }

    case "maxpool2dNode":
    case "avgpool2dNode":
      const poolKernel = Number(nodeData.kernel_size || 2)
      const poolStride = Number(nodeData.stride || poolKernel)
      const poolPadding = Number(nodeData.padding || 0)
      const poolDilation = nodeData.dilation || 1

      if (inputShape.height && inputShape.width) {
        if (poolStride === 0) return inputShape // Avoid division by zero

        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * poolPadding - poolDilation * (poolKernel - 1) - 1) / poolStride + 1,
        )
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * poolPadding - poolDilation * (poolKernel - 1) - 1) / poolStride + 1,
        )
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return inputShape

    case "maxpool1dNode":
    case "avgpool1dNode":
      const pool1dKernel = nodeData.kernel_size || 2
      const pool1dStride = nodeData.stride || pool1dKernel
      const pool1dPadding = nodeData.padding || 0
      const dilation1d = nodeData.dilation || 1

      if (inputShape.length) {
        const newLength = Math.floor(
          (Number(inputShape.length) + 2 * pool1dPadding - dilation1d * (pool1dKernel - 1) - 1) /
            pool1dStride +
            1,
        )
        return {
          channels: inputShape.channels,
          length: isNaN(newLength) ? "dynamic" : Math.max(1, newLength),
        }
      }
      return inputShape

    case "maxpool3dNode":
    case "avgpool3dNode":
      const pool3dKernel = nodeData.kernel_size || 2
      const pool3dStride = nodeData.stride || pool3dKernel
      const pool3dPadding = nodeData.padding || 0
      const dilation3d = nodeData.dilation || 1

      if (inputShape.depth && inputShape.height && inputShape.width) {
        const newDepth = Math.floor(
          (Number(inputShape.depth) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        )
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        )
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        )
        return {
          channels: inputShape.channels,
          depth: isNaN(newDepth) ? "dynamic" : Math.max(1, newDepth),
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return inputShape

    case "adaptivemaxpool1dNode":
      return {
        channels: inputShape.channels,
        length: nodeData.output_size || 1,
      }

    case "adaptivemaxpool3dNode":
      const output3d = nodeData.output_size || [1, 1, 1]
      return {
        channels: inputShape.channels,
        depth: output3d[0],
        height: output3d[1],
        width: output3d[2],
      }

    case "fractionalmaxpool2dNode":
      const outputRatio = nodeData.output_ratio || 0.5
      if (inputShape.height && inputShape.width) {
        const newHeight = Math.floor(Number(inputShape.height) * outputRatio)
        const newWidth = Math.floor(Number(inputShape.width) * outputRatio)
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : newHeight,
          width: isNaN(newWidth) ? "dynamic" : newWidth,
        }
      }
      return inputShape

    case "lppool2dNode":
      const lpKernel = Number(nodeData.kernel_size || 2)
      const lpStride = Number(nodeData.stride || lpKernel)
      const lpPadding = 0 // LPPool2d does not have padding

      if (inputShape.height && inputShape.width) {
        if (lpStride === 0) return inputShape

        const newHeight = Math.floor((Number(inputShape.height) + 2 * lpPadding - lpKernel) / lpStride + 1)
        const newWidth = Math.floor((Number(inputShape.width) + 2 * lpPadding - lpKernel) / lpStride + 1)
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return inputShape

    case "flattenNode":
      const start_dim = nodeData.start_dim ?? 0
      const end_dim = nodeData.end_dim ?? -1

      const orderedDims: (keyof TensorShape)[] = [
        "channels",
        "depth",
        "height",
        "width",
        "sequence",
        "length",
        "features",
      ]
      const dimsWithValue = orderedDims.filter(d => inputShape[d] !== undefined)

      if (dimsWithValue.length === 0) {
        return { features: "dynamic" }
      }

      const realEndDim = end_dim === -1 ? dimsWithValue.length - 1 : end_dim

      const output: Partial<TensorShape> = {}
      let flattenedDim: number | "dynamic" = 1
      let firstFlattenedDimName: keyof TensorShape | undefined

      for (let i = 0; i < dimsWithValue.length; i++) {
        const dimName = dimsWithValue[i]
        if (i >= start_dim && i <= realEndDim) {
          if (!firstFlattenedDimName) {
            firstFlattenedDimName = dimName
          }
          const val = inputShape[dimName]
          if (val === "dynamic" || flattenedDim === "dynamic") {
            flattenedDim = "dynamic"
          } else if (val !== undefined) {
            flattenedDim *= Number(val)
          }
        } else {
          ;(output as any)[dimName] = inputShape[dimName]
        }
      }

      if (firstFlattenedDimName) {
        const isFullFlatten = start_dim === 0 && realEndDim === dimsWithValue.length - 1
        const newDimName = isFullFlatten ? "features" : firstFlattenedDimName
        ;(output as any)[newDimName] = flattenedDim
      } else {
        return inputShape
      }

      return output as TensorShape

    case "adaptiveavgpool2dNode":
      const outputSize = nodeData.output_size;
      if (Array.isArray(outputSize) && outputSize.length === 2) {
        return {
          channels: inputShape.channels,
          height: outputSize[0] ?? 1,
          width: outputSize[1] ?? 1,
        };
      }
      return {
        channels: inputShape.channels,
        height: 1,
        width: 1,
      };

    case "multiheadattentionNode":
      return {
        sequence: inputShape.sequence,
        features: nodeData.embed_dim || 128,
      }

    case "transformerencoderlayerNode": // Assumes (seq, features)
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
      const concatDim = nodeData.dim ?? 0;
      if (inputShapes.length === 0) return {};

      const firstShape = inputShapes[0];
      if (!firstShape) return {};
      
      const orderedDimsCalc = getOrderedDimensions(firstShape);

      if (concatDim >= orderedDimsCalc.length) {
          return firstShape;
      }

      const targetDim = orderedDimsCalc[concatDim];

      const concatOutputShape = { ...firstShape };

      let totalDimSize: number | "dynamic" = 0;
      for (const shape of inputShapes) {
        if (!shape) continue;
        const dimSize = shape[targetDim];
        if (dimSize === "dynamic" || totalDimSize === "dynamic") {
          totalDimSize = "dynamic";
          break;
        }
        if (dimSize !== undefined) {
          totalDimSize += Number(dimSize);
        }
      }

      (concatOutputShape as any)[targetDim] = totalDimSize;
      return concatOutputShape

    case "separableconv2dNode":
      const sepPadding = nodeData.padding || 0
      const sepStride = nodeData.stride || 1
      const sepKernel = nodeData.kernel_size || 3
      const sepDilation = nodeData.dilation || 1

      if (inputShape.height && inputShape.width) {
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * sepPadding - sepDilation * (sepKernel - 1) - 1) / sepStride + 1,
        )
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * sepPadding - sepDilation * (sepKernel - 1) - 1) / sepStride + 1,
        )
        return {
          channels: nodeData.out_channels || 64,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        }
      }
      return { ...inputShape, channels: nodeData.out_channels || 64 }

    default:
      return {
        channels: inputShape.channels,
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
  if (!shape || Object.keys(shape).length === 0) {
    return "[?]";
  }

  const typedShape = shape as TensorShape;

  const dimOrder: (keyof TensorShape)[] = [
    "channels",
    "depth",
    "height",
    "width",
    "sequence",
    "length",
    "features",
  ];

  const parts = dimOrder
    .filter(dim => typedShape[dim] !== undefined)
    .map(dim => {
      const val = typedShape[dim];
      return val === "dynamic" ? "?" : `${val}`;
    });

  if (parts.length === 0) {
    const allValues = Object.values(typedShape).filter(v => v !== undefined);
    if (allValues.length > 0) {
      return `[${allValues.map(v => (v === "dynamic" ? "?" : v)).join(", ")}]`;
    }
    return "[?]";
  }

  return `[${parts.join(", ")}]`;
}

export function inferDynamicShape(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): TensorShape {
  if (!inputShapes || inputShapes.length === 0) {
    return calculateOutputShape(nodeType, [{}], nodeData)
  }

  // For operations that can work with multiple inputs, pass all shapes
  if (nodeType === "addNode" || nodeType === "concatenateNode") {
    return calculateOutputShape(nodeType, inputShapes, nodeData)
  }

  // For most operations, use the first input shape
  return calculateOutputShape(nodeType, [inputShapes[0]], nodeData)
}
