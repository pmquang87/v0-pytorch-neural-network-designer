import { type GraphIR, type GraphNode, type GraphEdge, PYTORCH_LAYER_MANIFEST } from "./types"

export class ModelGenerator {
  private readonly nodes: GraphNode[]
  private readonly edges: GraphEdge[]
  private readonly inputNode: GraphNode | null = null

  constructor(graph: GraphIR) {
    this.nodes = graph.nodes
    this.edges = graph.edges
    this.inputNode = this.nodes.find((node) => node.type === "inputNode") || null
  }

  validateGraph(): { valid: boolean; error?: string } {
    if (!this.inputNode) {
      return { valid: false, error: "Graph must contain an input node" }
    }

    const visited = new Set<string>()
    const recursionStack = new Set<string>()

    const hasCycle = (nodeId: string): boolean => {
      if (recursionStack.has(nodeId)) return true
      if (visited.has(nodeId)) return false

      visited.add(nodeId)
      recursionStack.add(nodeId)

      const outgoingEdges = this.edges.filter((edge) => edge.source === nodeId)
      for (const edge of outgoingEdges) {
        if (hasCycle(edge.target)) return true
      }

      recursionStack.delete(nodeId)
      return false
    }

    for (const node of this.nodes) {
      if (hasCycle(node.id)) {
        return { valid: false, error: "Graph contains cycles" }
      }
    }

    return { valid: true }
  }

  private buildGraphStructures(): { adjList: Map<string, string[]>; inDegree: Map<string, number> } {
    const inDegree = new Map<string, number>();
    const adjList = new Map<string, string[]>();

    for (const node of this.nodes) {
      inDegree.set(node.id, 0);
      adjList.set(node.id, []);
    }

    for (const edge of this.edges) {
      adjList.get(edge.source)?.push(edge.target);
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    }
    return { adjList, inDegree };
  }

  topologicalSort(): GraphNode[] {
    const { adjList, inDegree } = this.buildGraphStructures();

    const queue: string[] = []
    const result: GraphNode[] = []

    for (const [nodeId, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(nodeId)
      }
    }

    while (queue.length > 0) {
      const currentId = queue.shift()!
      const currentNode = this.nodes.find((n) => n.id === currentId)!
      result.push(currentNode)

      for (const neighborId of adjList.get(currentId) || []) {
        inDegree.set(neighborId, inDegree.get(neighborId)! - 1)
        if (inDegree.get(neighborId) === 0) {
          queue.push(neighborId)
        }
      }
    }

    return result
  }

  generateCode(): string {
    const validation = this.validateGraph()
    if (!validation.valid) {
      throw new Error(validation.error)
    }

    const sortedNodes = this.topologicalSort()
    const layerNodes = sortedNodes.filter((node) => node.type !== "inputNode" && node.type !== "outputNode")

    let code = `import torch\nimport torch.nn as nn\n`

    const hasSsmNode = this.nodes.some(node => node.type === 'ssmNode');
    if (hasSsmNode) {
        code += `\n\nclass SSM(nn.Module):\n    def __init__(self, d_model, d_state):\n        super().__init__()\n        self.d_model = d_model\n        self.d_state = d_state\n
        # Learnable parameters A, B, C\n        # A: State transition matrix\n        # B: Input to state matrix\n        # C: State to output matrix\n        self.A = nn.Parameter(torch.randn(d_state, d_state))\n        self.B = nn.Parameter(torch.randn(d_state, d_model))\n        self.C = nn.Parameter(torch.randn(d_model, d_state))\n
    def forward(self, x):\n        # x: input tensor of shape (batch_size, sequence_length, d_model)\n        batch_size, sequence_length, d_model = x.shape\n
        # Initialize the hidden state\n        h = torch.zeros(batch_size, self.d_state, device=x.device)\n
        outputs = []\n        for t in range(sequence_length):\n            # Get the input at the current time step\n            x_t = x[:, t, :]\n
            # Update the hidden state (linear recurrence)\n            h = torch.tanh(torch.matmul(self.A, h.unsqueeze(-1)).squeeze(-1) + torch.matmul(self.B, x_t.unsqueeze(-1)).squeeze(-1))\n
            # Compute the output at the current time step\n            y_t = torch.matmul(self.C, h.unsqueeze(-1)).squeeze(-1)\n
            outputs.append(y_t)\n        \n        return torch.stack(outputs, dim=1)\n`
    }

    code += `\n\nclass GeneratedModel(nn.Module):\n    def __init__(self):\n        super(GeneratedModel, self).__init__()\n        \n`

    for (const node of layerNodes) {
      const manifest = PYTORCH_LAYER_MANIFEST[node.type as keyof typeof PYTORCH_LAYER_MANIFEST]
      const layerName = this.sanitizeLayerName(node.id)

      if (node.type === "adaptiveavgpool2dNode") {
        const outputSize = JSON.stringify(node.data.output_size)
        code += `        self.${layerName} = nn.AdaptiveAvgPool2d(output_size=${outputSize})\n`
      } else if (node.type === "flattenNode") {
        const startDim = node.data.start_dim ?? 1
        code += `        self.${layerName} = nn.Flatten(start_dim=${startDim})\n`
      } else if (node.type === "timeDistributedLinearNode") {
        const in_features = node.data.in_features
        const out_features = node.data.out_features
        code += `        self.${layerName} = nn.Linear(in_features=${in_features}, out_features=${out_features})\n`
      } else if (node.type === "siluNode") {
        code += `        self.${layerName} = nn.SiLU()\n`
      } else if (node.type === "ssmNode") {
        const { d_model, d_state } = node.data;
        code += `        self.${layerName} = SSM(d_model=${d_model}, d_state=${d_state})\n`
      } else if (manifest && manifest.className) {
        if (node.type === "depthwiseconv2dNode") {
          const params = this.buildParameterString(node, manifest.params)
          const groups = node.data.in_channels || node.data.groups || 1
          code += `        self.${layerName} = ${manifest.className}(${params}, groups=${groups})\n`
        } else if (node.type === "separableconv2dNode") {
          const inChannels = node.data.in_channels || 32
          const outChannels = node.data.out_channels || 64
          const kernelSize = node.data.kernel_size || 3
          const stride = node.data.stride || 1
          const padding = node.data.padding || 1

          code += `        # Separable convolution: depthwise + pointwise\n`
          code += `        self.${layerName}_depthwise = nn.Conv2d(${inChannels}, ${inChannels}, kernel_size=${kernelSize}, stride=${stride}, padding=${padding}, groups=${inChannels})\n`
          code += `        self.${layerName}_pointwise = nn.Conv2d(${inChannels}, ${outChannels}, kernel_size=1, stride=1, padding=0)\n`
        } else {
          const params = this.buildParameterString(node, manifest.params)
          code += `        self.${layerName} = ${manifest.className}(${params})\n`
        }
      }
    }

    code += `\n    def forward(self, x):\n`

    const forwardLines = this.generateForwardPass(sortedNodes)
    for (const line of forwardLines) {
      code += `        ${line}\n`
    }

    code += `        return x\n`

    if (this.inputNode?.data) {
      const { channels, height, width } = this.inputNode.data
      const shape = [1, channels, height, width].filter(d => d !== undefined)
      if (shape.length > 1) {
          const shapeString = JSON.stringify(shape).slice(1, -1)
          code += `\n\n\n# Usage example:\n# model = GeneratedModel()\n# input_tensor = torch.randn(1, ${shapeString})\n# output = model(input_tensor)\n# print(f\"Input shape: {input_tensor.shape}\")\n# print(f\"Output shape: {output.shape}\")\n`
      }
    }

    return code
  }

  private buildParameterString(node: GraphNode, paramNames: string[]): string {
    const params: string[] = []
    for (const paramName of paramNames) {
      const value = node.data[paramName]
      if (value !== undefined && value !== null) {
        if (Array.isArray(value)) {
          params.push(`${paramName}=${JSON.stringify(value)}`)
        } else if (typeof value === "string") {
          params.push(`${paramName}=\"${value}\"`)
        } else {
          params.push(`${paramName}=${value}`)
        }
      }
    }
    return params.join(", ")
  }

  private sanitizeLayerName(nodeId: string): string {
    return nodeId.replace(/[^a-zA-Z0-9_]/g, "_")
  }

  private generateForwardPass(sortedNodes: GraphNode[]): string[] {
    const lines: string[] = []
    const nodeOutputs = new Map<string, string>()

    if (this.inputNode) {
      nodeOutputs.set(this.inputNode.id, "x")
    }

    for (const node of sortedNodes) {
      if (node.type === "inputNode" || node.type === "outputNode") continue

      const layerName = this.sanitizeLayerName(node.id)
      const inputEdges = this.edges.filter((edge) => edge.target === node.id)

      if (inputEdges.length === 0) {
        continue
      }

      if (node.type === "addNode") {
        const inputVars = inputEdges.map((edge) => nodeOutputs.get(edge.source) || "x")
        const outputVar = `x_${layerName}`
        if (inputVars.length >= 2) {
          lines.push(`${outputVar} = ${inputVars.join(" + ")}`)
        } else {
          lines.push(`${outputVar} = ${inputVars[0]}  # Single input to add node`)
        }
        nodeOutputs.set(node.id, outputVar)
      } else if (node.type === "concatenateNode") {
        const inputVars = inputEdges.map((edge) => nodeOutputs.get(edge.source) || "x")
        const outputVar = `x_${layerName}`
        const dim = node.data.dim || 1

        if (inputVars.length === 2) {
          const decoderVar = inputVars[0]
          const skipVar = inputVars[1]
          const croppedSkipVar = `${skipVar}_cropped`

          lines.push(`# Crop skip connection to match decoder spatial dimensions`)
          lines.push(`decoder_h, decoder_w = ${decoderVar}.shape[2], ${decoderVar}.shape[3]`)
          lines.push(`skip_h, skip_w = ${skipVar}.shape[2], ${skipVar}.shape[3]`)
          lines.push(`if skip_h != decoder_h or skip_w != decoder_w:`)
          lines.push(`    start_h = (skip_h - decoder_h) // 2`)
          lines.push(`    start_w = (skip_w - decoder_w) // 2`)
          lines.push(`    ${croppedSkipVar} = ${skipVar}[:, :, start_h:start_h+decoder_h, start_w:start_w+decoder_w]`)
          lines.push(`else:`)
          lines.push(`    ${croppedSkipVar} = ${skipVar}`)

          lines.push(`${outputVar} = torch.cat([${decoderVar}, ${croppedSkipVar}], dim=${dim})`)
        } else {
          lines.push(`${outputVar} = torch.cat([${inputVars.join(", ")}], dim=${dim})`)
        }

        nodeOutputs.set(node.id, outputVar)
      } else if (node.type === "separableconv2dNode") {
        const inputVar = nodeOutputs.get(inputEdges[0].source) || "x"
        const outputVar = `x_${layerName}`
        const tempVar = `x_${layerName}_temp`
        lines.push(`${tempVar} = self.${layerName}_depthwise(${inputVar})`)
        lines.push(`${outputVar} = self.${layerName}_pointwise(${tempVar})`)
        nodeOutputs.set(node.id, outputVar)
      } else if (node.type === 'multiheadattentionNode') {
        const queryVar = nodeOutputs.get(inputEdges.find(e => e.targetHandle === 'query')?.source || '') || 'x'
        const keyVar = nodeOutputs.get(inputEdges.find(e => e.targetHandle === 'key')?.source || '') || 'x'
        const valueVar = nodeOutputs.get(inputEdges.find(e => e.targetHandle === 'value')?.source || '') || 'x'
        const outputVar = `x_${layerName}`
        lines.push(`${outputVar}, _ = self.${layerName}(${queryVar}, ${keyVar}, ${valueVar})`)
        nodeOutputs.set(node.id, outputVar)
      } else if (inputEdges.length === 1) {
        const inputVar = nodeOutputs.get(inputEdges[0].source) || "x"
        const outputVar = `x_${layerName}`
        lines.push(`${outputVar} = self.${layerName}(${inputVar})`)
        nodeOutputs.set(node.id, outputVar)
      } else {
        const inputVars = inputEdges.map((edge) => nodeOutputs.get(edge.source) || "x")
        const outputVar = `x_${layerName}`
        lines.push(`# Warning: Multiple inputs to regular layer - using first input`)
        lines.push(`${outputVar} = self.${layerName}(${inputVars[0]})`)
        nodeOutputs.set(node.id, outputVar)
      }
    }

    const lastNode = sortedNodes[sortedNodes.length - 1];
    let finalOutputVar: (string | undefined) = undefined;

    if (lastNode && lastNode.type === 'outputNode') {
        const finalEdge = this.edges.find(edge => edge.target === lastNode.id);
        if (finalEdge) {
            finalOutputVar = nodeOutputs.get(finalEdge.source);
        }
    } else if (lastNode) {
        finalOutputVar = nodeOutputs.get(lastNode.id);
    }

    if (finalOutputVar && finalOutputVar !== 'x') {
        lines.push(`x = ${finalOutputVar}`);
    }

    return lines
  }
}
