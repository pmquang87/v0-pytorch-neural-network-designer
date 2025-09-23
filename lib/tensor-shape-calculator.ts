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
  inputShapes: (TensorShape | undefined)[], // Allow undefined for disconnected inputs
  nodeData: any,
): ShapeValidationResult {
  switch (nodeType) {
    case "addNode":
    case "multiplyNode": {
      const definedInputs = inputShapes
        .map((shape, index) => ({ shape, index }))
        .filter(item => item.shape && Object.keys(item.shape).length > 0);

      if (definedInputs.length < 2) {
        return { isValid: true }; // Not enough inputs to have a mismatch
      }

      const allKeys = new Set<keyof TensorShape>();
      definedInputs.forEach(({ shape }) => {
        if (shape) {
          (Object.keys(shape) as (keyof TensorShape)[]).forEach(key => {
            allKeys.add(key);
          });
        }
      });

      for (const key of allKeys) {
        const sizes = new Set<number>();
        const contributingInputs = new Map<number, number>(); // size -> input index

        for (const { shape, index } of definedInputs) {
          const val = shape?.[key];
          // Treat undefined dimension as 1 for broadcasting
          const dim = val === undefined ? 1 : val;

          if (typeof dim === "number" && dim !== 1) {
            sizes.add(dim);
            if (!contributingInputs.has(dim)) {
              // Use the original index + 1 for user-facing handle ID
              contributingInputs.set(dim, index + 1);
            }
          }
        }

        if (sizes.size > 1) {
          const nodeName = nodeType === "addNode" ? "Add" : "Multiply";
          const sizeArray = Array.from(sizes).sort((a, b) => a - b);
          const errorDetails = sizeArray.map(s => `${s} (from input ${contributingInputs.get(s)})`).join(" vs ");
          return {
            isValid: false,
            error: `${nodeName} error: Incompatible sizes for broadcasting on dimension '${key}'. Got ${errorDetails}.`,
          };
        }
      }
      break;
    }

    case "concatenateNode": {
      const definedInputs = inputShapes
        .map((shape, index) => ({ shape, index }))
        .filter(item => item.shape);

      if (definedInputs.length < 2) {
        return { isValid: true };
      }

      const firstInput = definedInputs[0];
      if (!firstInput.shape) break;
      const firstDims = getOrderedDimensions(firstInput.shape);

      // Check that all inputs have the same dimension keys
      for (let i = 1; i < definedInputs.length; i++) {
        const currentInput = definedInputs[i];
        if (!currentInput.shape) continue;
        const currentDims = getOrderedDimensions(currentInput.shape);
        if (firstDims.length !== currentDims.length || !firstDims.every((dim, idx) => dim === currentDims[idx])) {
          return {
            isValid: false,
            error: `Concatenation error: All inputs must have the same dimensions. Input ${firstInput.index + 1} has [${firstDims.join(
              ", ",
            )}], but input ${currentInput.index + 1} has [${currentDims.join(", ")}].`,
          };
        }
      }

      const dim = nodeData.dim ?? 1;
      if (dim === 0) {
        return { isValid: false, error: "Concatenation on dim=0 (batch dimension) is not supported." };
      }
      const codeDim = dim - 1;

      if (codeDim < 0 || codeDim >= firstDims.length) {
        return {
          isValid: false,
          error: `Invalid dimension ${dim} for a tensor with ${firstDims.length} dimensions.`,
        };
      }

      const concatDimName = firstDims[codeDim];

      // Check that all non-concatenated dimensions match
      for (const dimName of firstDims) {
        if (dimName === concatDimName) continue;

        const dimensionValues = new Map<number, number[]>(); // size -> list of input indices
        for (const { shape, index } of definedInputs) {
          const val = shape?.[dimName];
          if (typeof val === "number") {
            if (!dimensionValues.has(val)) {
              dimensionValues.set(val, []);
            }
            dimensionValues.get(val)!.push(index + 1);
          }
        }

        if (dimensionValues.size > 1) {
          const errorDetails = Array.from(dimensionValues.entries())
            .map(([size, inputs]) => `${size} (on input${inputs.length > 1 ? "s" : ""} ${inputs.join(", ")})`)
            .join(" vs ");
          return {
            isValid: false,
            error: `Concatenation error: Incompatible sizes for dimension '${dimName}'. All non-concatenated dimensions must match. Found: ${errorDetails}.`,
          };
        }
      }
      break;
    }

    case "reshapeNode": {
      const targetShapeStr = (nodeData.targetShape || "").trim();
      if (!targetShapeStr) {
        return { isValid: true }; // No shape to validate yet
      }
      const inputShape = inputShapes[0];
      if (!inputShape || Object.keys(inputShape).length === 0) {
        return { isValid: true, warning: "Connect an input to validate reshape." };
      }

      let targetDims: number[];
      try {
        const jsonFriendlyStr = targetShapeStr.replace(/'/g, '"').replace(/,(\s*?)]/g, ']');
        const parsed = JSON.parse(jsonFriendlyStr);
        if (!Array.isArray(parsed) || !parsed.every(d => typeof d === 'number')) {
          return { isValid: false, error: "Invalid format. Shape must be a list of integers, e.g., [-1, 784]." };
        }
        targetDims = parsed;
      } catch (e) {
        return { isValid: false, error: "Invalid format. Use a list of integers like [-1, 784]." };
      }

      if (targetDims.filter(d => d === -1).length > 1) {
        return { isValid: false, error: "Invalid shape: can only have one '-1' dimension." };
      }

      const inputDimValues = Object.values(inputShape).filter(v => typeof v === 'number' || v === 'dynamic');
      if (inputDimValues.some(v => v === "dynamic")) {
        if (targetDims.includes(-1)) {
          return { isValid: true, warning: "Cannot infer '-1' dimension when input shape is dynamic." };
        }
        return { isValid: true };
      }

      const totalInputSize = inputDimValues.filter((v): v is number => typeof v === 'number').reduce((acc, val) => acc * val, 1);
      const inferredDimIndex = targetDims.indexOf(-1);

      if (inferredDimIndex !== -1) {
        const productOfKnownDims = targetDims.reduce((acc, val) => (val !== -1 ? acc * val : acc), 1);
        if (productOfKnownDims > 0 && totalInputSize % productOfKnownDims !== 0) {
          return { isValid: false, error: `Cannot reshape tensor of size ${totalInputSize} into shape with product ${productOfKnownDims}.` };
        }
      } else {
        const totalOutputSize = targetDims.reduce((acc, val) => acc * val, 1);
        if (totalInputSize !== totalOutputSize) {
          return { isValid: false, error: `Shape mismatch. Input has ${totalInputSize} elements, target has ${totalOutputSize}.` };
        }
      }

      return { isValid: true };
    }

    case "gruNode":
    case "lstmNode":
    case "rnnNode": {
      const inputSize = nodeData.input_size;
      const inputFeatures = inputShapes[0]?.features || inputShapes[0]?.width;
      if (inputSize && inputFeatures && inputSize !== inputFeatures && inputFeatures !== "dynamic") {
        const nodeName = nodeType.replace('Node', '').toUpperCase();
        return {
          isValid: false,
          error: `${nodeName} layer expects input_size=${inputSize}, but received ${inputFeatures}.`,
        };
      }
      break;
    }

    case "linearNode":
      const in_features = nodeData.in_features;
      const input_features = inputShapes[0]?.features || inputShapes[0]?.width || inputShapes[0]?.channels;
      if (in_features && input_features && in_features !== input_features && input_features !== "dynamic") {
        return {
          isValid: false,
          error: `Input features ${input_features} do not match layer\'s in_features ${in_features}`,
        };
      }
      break;

    case "timeDistributedLinearNode":
      const td_in_features = nodeData.in_features;
      const td_input_features = inputShapes[0]?.features || inputShapes[0]?.width;
      if (td_in_features && td_input_features && td_in_features !== td_input_features && td_input_features !== "dynamic") {
        return {
          isValid: false,
          error: `Input features ${td_input_features} do not match layer\'s in_features ${td_in_features}`,
        };
      }
      break;

    case "multiheadattentionNode":
      const embedDim = nodeData.embed_dim || 128;
      const inferredFeatureDim = inputShapes[0]?.features || inputShapes[0]?.width || inputShapes[0]?.channels;
      if (inferredFeatureDim !== undefined && inferredFeatureDim !== "dynamic" && inferredFeatureDim !== embedDim) {
        return {
          isValid: false,
          error: `MultiheadAttention expects input dimension ${embedDim}, got ${inferredFeatureDim}`,
        };
      }
      break;
  }

  return { isValid: true };
}

export function calculateOutputShape(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): TensorShape {
  const inputShape = inputShapes[0] || {};
  switch (nodeType) {
    case "inputNode":
      const outputShape: TensorShape = {};
      const possibleDims: (keyof TensorShape)[] = [
        "channels",
        "depth",
        "height",
        "width",
        "sequence",
        "length",
        "features",
      ];
      for (const dim of possibleDims) {
        if (nodeData[dim] !== undefined) {
          (outputShape as any)[dim] = nodeData[dim];
        }
      }
      return outputShape;

    case "outputNode":
      // Output node is a sink; pass through the incoming shape unchanged
      return { ...inputShape };

    case "conv2dNode":
    case "depthwiseconv2dNode":
      const padding = parseTuple(nodeData.padding, 0);
      const stride = parseTuple(nodeData.stride, 1);
      const kernel = parseTuple(nodeData.kernel_size, 3);
      const dilation = parseTuple(nodeData.dilation, 1);

      if (inputShape.height && inputShape.width && stride[0] > 0 && stride[1] > 0) {
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0] + 1,
        );
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1] + 1,
        );
        return {
          channels: nodeData.out_channels || 32,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 };

    case "conv1dNode":
      const conv1dPadding = nodeData.padding || 0;
      const conv1dStride = nodeData.stride || 1;
      const conv1dKernel = nodeData.kernel_size || 3;
      const conv1dDilation = nodeData.dilation || 1;

      if (inputShape.length) {
        const newLength = Math.floor(
          (Number(inputShape.length) + 2 * conv1dPadding - conv1dDilation * (conv1dKernel - 1) - 1) /
            conv1dStride +
            1,
        );
        return {
          channels: nodeData.out_channels || 32,
          length: isNaN(newLength) ? "dynamic" : Math.max(1, newLength),
        };
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 };

    case "conv3dNode":
      const conv3dPadding = nodeData.padding || 0;
      const conv3dStride = nodeData.stride || 1;
      const conv3dKernel = nodeData.kernel_size || 3;
      const conv3dDilation = nodeData.dilation || 1;

      if (inputShape.depth && inputShape.height && inputShape.width) {
        const newDepth = Math.floor(
          (Number(inputShape.depth) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        );
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        );
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * conv3dPadding - conv3dDilation * (conv3dKernel - 1) - 1) /
            conv3dStride +
            1,
        );
        return {
          channels: nodeData.out_channels || 32,
          depth: isNaN(newDepth) ? "dynamic" : Math.max(1, newDepth),
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return { ...inputShape, channels: nodeData.out_channels || 32 };

    case "convtranspose2dNode":
      const transStride = nodeData.stride || 2;
      const transKernel = nodeData.kernel_size || 2;
      const transPadding = nodeData.padding || 0;
      const outputPadding = nodeData.output_padding || 0;
      const transDilation = nodeData.dilation || 1;

      if (inputShape.height && inputShape.width) {
        const newHeight =
          (Number(inputShape.height) - 1) * transStride -
          2 * transPadding +
          transDilation * (transKernel - 1) +
          outputPadding +
          1;
        const newWidth =
          (Number(inputShape.width) - 1) * transStride -
          2 * transPadding +
          transDilation * (transKernel - 1) +
          outputPadding +
          1;
        return {
          channels: nodeData.out_channels || 16,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return { ...inputShape, channels: nodeData.out_channels || 16 };

    case "linearNode":
      return {
        features: nodeData.out_features || 64,
      };

    case "timeDistributedLinearNode":
      return {
        ...inputShape,
        features: nodeData.out_features || 64,
        width: nodeData.out_features || 64, // Also update width for (C, H, W) format
      };

    case "maxpool2dNode":
    case "avgpool2dNode":
      const poolKernel = Number(nodeData.kernel_size || 2);
      const poolStride = Number(nodeData.stride || poolKernel);
      const poolPadding = Number(nodeData.padding || 0);
      const poolDilation = nodeData.dilation || 1;

      if (inputShape.height && inputShape.width) {
        if (poolStride === 0) return inputShape; // Avoid division by zero

        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * poolPadding - poolDilation * (poolKernel - 1) - 1) / poolStride + 1,
        );
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * poolPadding - poolDilation * (poolKernel - 1) - 1) / poolStride + 1,
        );
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return inputShape;

    case "maxpool1dNode":
    case "avgpool1dNode":
      const pool1dKernel = nodeData.kernel_size || 2;
      const pool1dStride = nodeData.stride || pool1dKernel;
      const pool1dPadding = nodeData.padding || 0;
      const dilation1d = nodeData.dilation || 1;

      if (inputShape.length) {
        const newLength = Math.floor(
          (Number(inputShape.length) + 2 * pool1dPadding - dilation1d * (pool1dKernel - 1) - 1) /
            pool1dStride +
            1,
        );
        return {
          channels: inputShape.channels,
          length: isNaN(newLength) ? "dynamic" : Math.max(1, newLength),
        };
      }
      return inputShape;

    case "maxpool3dNode":
    case "avgpool3dNode":
      const pool3dKernel = nodeData.kernel_size || 2;
      const pool3dStride = nodeData.stride || pool3dKernel;
      const pool3dPadding = nodeData.padding || 0;
      const dilation3d = nodeData.dilation || 1;

      if (inputShape.depth && inputShape.height && inputShape.width) {
        const newDepth = Math.floor(
          (Number(inputShape.depth) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        );
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        );
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * pool3dPadding - dilation3d * (pool3dKernel - 1) - 1) /
            pool3dStride +
            1,
        );
        return {
          channels: inputShape.channels,
          depth: isNaN(newDepth) ? "dynamic" : Math.max(1, newDepth),
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return inputShape;

    case "adaptivemaxpool1dNode":
      return {
        channels: inputShape.channels,
        length: nodeData.output_size || 1,
      };

    case "adaptivemaxpool3dNode":
      const output3d = nodeData.output_size || [1, 1, 1];
      return {
        channels: inputShape.channels,
        depth: output3d[0],
        height: output3d[1],
        width: output3d[2],
      };

    case "fractionalmaxpool2dNode":
      const outputRatio = nodeData.output_ratio || 0.5;
      if (inputShape.height && inputShape.width) {
        const newHeight = Math.floor(Number(inputShape.height) * outputRatio);
        const newWidth = Math.floor(Number(inputShape.width) * outputRatio);
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : newHeight,
          width: isNaN(newWidth) ? "dynamic" : newWidth,
        };
      }
      return inputShape;

    case "lppool2dNode":
      const lpKernel = Number(nodeData.kernel_size || 2);
      const lpStride = Number(nodeData.stride || lpKernel);
      const lpPadding = 0; // LPPool2d does not have padding

      if (inputShape.height && inputShape.width) {
        if (lpStride === 0) return inputShape;

        const newHeight = Math.floor((Number(inputShape.height) + 2 * lpPadding - lpKernel) / lpStride + 1);
        const newWidth = Math.floor((Number(inputShape.width) + 2 * lpPadding - lpKernel) / lpStride + 1);
        return {
          channels: inputShape.channels,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return inputShape;

    case "flattenNode": {
      const start_dim_ui = nodeData.start_dim ?? 1;
      const end_dim_ui = nodeData.end_dim ?? -1;
      const start_dim = start_dim_ui - 1;

      const standardDims = getOrderedDimensions(inputShape);
      const genericKeys = Object.keys(inputShape)
        .filter(k => k.startsWith("dim"))
        .sort((a, b) => {
          const numA = parseInt(a.replace("dim", ""), 10);
          const numB = parseInt(b.replace("dim", ""), 10);
          return isNaN(numA) || isNaN(numB) ? a.localeCompare(b) : numA - numB;
        });

      const dimsToProcess = standardDims.length > 0 ? standardDims : (genericKeys as (keyof TensorShape)[]);
      const isGenericInput = standardDims.length === 0 && genericKeys.length > 0;

      if (dimsToProcess.length === 0) {
        return { features: "dynamic" };
      }

      const realEndDim = end_dim_ui === -1 ? dimsToProcess.length - 1 : end_dim_ui - 1;

      if (start_dim >= dimsToProcess.length || realEndDim < 0 || start_dim > realEndDim) {
        return inputShape;
      }

      const output: Partial<TensorShape> = {};
      let flattenedDim: number | "dynamic" = 1;
      let firstFlattenedDimName: keyof TensorShape | undefined;

      for (let i = 0; i < dimsToProcess.length; i++) {
        const dimName = dimsToProcess[i];
        if (i >= start_dim && i <= realEndDim) {
          if (!firstFlattenedDimName) {
            firstFlattenedDimName = dimName;
          }
          const val = (inputShape as any)[dimName];
          if (val === "dynamic" || flattenedDim === "dynamic") {
            flattenedDim = "dynamic";
          } else if (val !== undefined) {
            flattenedDim *= Number(val);
          }
        } else {
          (output as any)[dimName] = (inputShape as any)[dimName];
        }
      }

      if (firstFlattenedDimName) {
        const isFullFlatten = start_dim === 0 && realEndDim === dimsToProcess.length - 1;
        const newDimName = isGenericInput || isFullFlatten ? "features" : firstFlattenedDimName;
        (output as any)[newDimName] = flattenedDim;
      } else {
        return inputShape;
      }

      return output as TensorShape;
    }

    case "reshapeNode": {
      const targetShapeStr = (nodeData.targetShape || "").trim();
      if (!inputShape || Object.keys(inputShape).length === 0 || !targetShapeStr) {
        return {};
      }

      let targetDims: number[];
      try {
        const jsonFriendlyStr = targetShapeStr.replace(/'/g, '"').replace(/,(\s*?)]/g, ']');
        targetDims = JSON.parse(jsonFriendlyStr);
        if (!Array.isArray(targetDims) || !targetDims.every(d => typeof d === 'number')) {
            return { error: "invalid" } as any;
        }
      } catch (e) {
        return { error: "invalid" } as any;
      }

      if (targetDims.filter(d => d === -1).length > 1) {
        return { error: "multiple -1s" } as any;
      }

      const inputDimValues = Object.values(inputShape).filter(v => typeof v === 'number' || v === 'dynamic');

      if (inputDimValues.some(v => v === "dynamic")) {
        const newShape: { [key: string]: string | number } = {};
        targetDims.forEach((dim, i) => {
          newShape[`dim${i}`] = dim === -1 ? "dynamic" : dim;
        });
        return newShape as any;
      }

      const totalInputSize = inputDimValues.filter((v): v is number => typeof v === 'number').reduce((acc, val) => acc * val, 1);
      const inferredDimIndex = targetDims.indexOf(-1);

      let finalDims: number[];
      if (inferredDimIndex !== -1) {
        const productOfKnownDims = targetDims.reduce((acc, val) => (val !== -1 ? acc * val : acc), 1);
        if (productOfKnownDims === 0 || totalInputSize % productOfKnownDims !== 0) {
          return { error: "mismatch" } as any;
        }
        const inferredDimValue = totalInputSize / productOfKnownDims;
        finalDims = targetDims.map(d => (d === -1 ? inferredDimValue : d));
      } else {
        const totalOutputSize = targetDims.reduce((acc, val) => acc * val, 1);
        if (totalInputSize !== totalOutputSize) {
            return { error: "mismatch" } as any;
        }
        finalDims = targetDims;
      }
      
      const newShape: { [key: string]: number } = {};
      finalDims.forEach((dim, i) => {
        newShape[`dim${i}`] = dim;
      });
      return newShape as any;
    }

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

    case "gruNode":
    case "lstmNode":
    case "rnnNode":
      const sequenceLength = inputShape.sequence || inputShape.channels;
      return {
        sequence: sequenceLength,
        features: nodeData.hidden_size,
      };

    case "multiheadattentionNode":
      // Preserve spatial/sequence layout used in examples: (C=1, H=sequence, W=embed)
      if (inputShape.width !== undefined) {
        return {
          ...inputShape,
          width: nodeData.embed_dim || 128,
        };
      }
      // Fallback to (sequence, features) representation
      return {
        sequence: inputShape.sequence,
        features: nodeData.embed_dim || 128,
      };

    case "transformerencoderlayerNode": // Assumes (seq, features)
    case "transformerdecoderlayerNode":
      return inputShape; // Transformer layers preserve input shape

    case "layernormNode":
      return inputShape;

    case "groupnormNode":
      return inputShape;

    case "instancenorm1dNode":
    case "instancenorm2dNode":
    case "instancenorm3dNode":
      return inputShape;

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
      return inputShape;

    case "batchnorm1dNode":
    case "batchnorm2dNode":
      return inputShape;

    case "addNode":
    case "multiplyNode": {
      const definedShapes = inputShapes.filter(s => s && Object.keys(s).length > 0);
      if (definedShapes.length === 0) return {};

      const broadcastShape: TensorShape = {};
      const allKeys = new Set<keyof TensorShape>();
      definedShapes.forEach(s => {
        if (s) {
          (Object.keys(s) as (keyof TensorShape)[]).forEach(k => allKeys.add(k));
        }
      });

      for (const key of allKeys) {
        let outputDimSize: number | "dynamic" = 1;
        for (const shape of definedShapes) {
          const size = shape?.[key];
          if (size === "dynamic") {
            outputDimSize = "dynamic";
            break;
          }
          if (typeof size === "number") {
            if (outputDimSize !== "dynamic") {
              outputDimSize = Math.max(outputDimSize, size);
            }
          }
        }
        broadcastShape[key] = outputDimSize;
      }
      return broadcastShape;
    }

    case "concatenateNode":
      const concatDimUi = nodeData.dim ?? 1;
      if (inputShapes.length === 0) return {};

      const firstConcatShape = inputShapes.find(s => s && Object.keys(s).length > 0);
      if (!firstConcatShape) return {};

      const orderedConcatDimsCalc = getOrderedDimensions(firstConcatShape);
      const concatDim = concatDimUi - 1; // UI is 1-based, code is 0-based

      if (concatDim < 0 || concatDim >= orderedConcatDimsCalc.length) {
        return firstConcatShape;
      }

      const targetConcatDim = orderedConcatDimsCalc[concatDim];

      const concatOutputShape = { ...firstConcatShape };

      let totalDimSize: number | "dynamic" = 0;
      for (const shape of inputShapes) {
        if (!shape) continue;
        const dimSize = shape[targetConcatDim];
        if (dimSize === "dynamic" || totalDimSize === "dynamic") {
          totalDimSize = "dynamic";
          break;
        }
        if (dimSize !== undefined) {
          totalDimSize += Number(dimSize);
        }
      }

      (concatOutputShape as any)[targetConcatDim] = totalDimSize;
      return concatOutputShape;

    case "separableconv2dNode":
      const sepPadding = parseTuple(nodeData.padding, 0);
      const sepStride = parseTuple(nodeData.stride, 1);
      const sepKernel = parseTuple(nodeData.kernel_size, 3);
      const sepDilation = parseTuple(nodeData.dilation, 1);

      if (inputShape.height && inputShape.width && sepStride[0] > 0 && sepStride[1] > 0) {
        const newHeight = Math.floor(
          (Number(inputShape.height) + 2 * sepPadding[0] - sepDilation[0] * (sepKernel[0] - 1) - 1) / sepStride[0] + 1,
        );
        const newWidth = Math.floor(
          (Number(inputShape.width) + 2 * sepPadding[1] - sepDilation[1] * (sepKernel[1] - 1) - 1) / sepStride[1] + 1,
        );
        return {
          channels: nodeData.out_channels || 64,
          height: isNaN(newHeight) ? "dynamic" : Math.max(1, newHeight),
          width: isNaN(newWidth) ? "dynamic" : Math.max(1, newWidth),
        };
      }
      return { ...inputShape, channels: nodeData.out_channels || 64 };

    default:
      return inputShape;
  }
}

export function formatTensorShape(shape: TensorShape | {} | undefined): string {
  if (!shape || Object.keys(shape).length === 0) {
    return "[?]";
  }

  if ('error' in shape) {
    return "[err]";
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

  if (parts.length > 0) {
    return `[${parts.join(", ")}]`;
  }

  // Fallback for generic shapes (e.g., from reshape)
  const genericKeys = Object.keys(typedShape).sort((a, b) => {
      const numA = parseInt(a.replace('dim', ''), 10);
      const numB = parseInt(b.replace('dim', ''), 10);
      if (!isNaN(numA) && !isNaN(numB)) {
          return numA - numB;
      }
      return a.localeCompare(b);
  });
  
  if (genericKeys.length > 0) {
    const allValues = genericKeys.map(key => (typedShape as any)[key]);
     if (allValues.some(v => typeof v !== 'number' && v !== 'dynamic')) {
        return "[err]"; // Contains non-numeric/dynamic values
    }
    return `[${allValues.map(v => (v === "dynamic" ? "?" : v)).join(", ")}]`;
  }

  return "[?]";
}

export function inferDynamicShape(
  nodeType: string,
  inputShapes: TensorShape[],
  nodeData: any,
): TensorShape {
  if (!inputShapes || inputShapes.length === 0) {
    return calculateOutputShape(nodeType, [{}], nodeData);
  }

  // For operations that can work with multiple inputs, pass all shapes
  if (nodeType === "addNode" || nodeType === "concatenateNode" || nodeType === "multiplyNode") {
    return calculateOutputShape(nodeType, inputShapes, nodeData);
  }

  // For most operations, use the first input shape
  return calculateOutputShape(nodeType, [inputShapes[0]], nodeData);
}
