import { type GraphIR, type GraphNode, type GraphEdge, PYTORCH_LAYER_MANIFEST } from "./types"

// Normalize a kernel/stride/padding value (number, [h, w] array, or "h,w"
// string) to an [h, w] pair so the arithmetic below never produces NaN.
function toScalarPair(value: any, defaultValue: number): [number, number] {
  if (typeof value === "number" && !isNaN(value)) return [value, value]
  if (Array.isArray(value)) {
    const nums = value.map(Number).filter((n) => !isNaN(n))
    if (nums.length >= 2) return [nums[0], nums[1]]
    if (nums.length === 1) return [nums[0], nums[0]]
  }
  if (typeof value === "string") {
    const nums = value
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !isNaN(n))
    if (nums.length >= 2) return [nums[0], nums[1]]
    if (nums.length === 1) return [nums[0], nums[0]]
  }
  return [defaultValue, defaultValue]
}

export interface VisualizationLayer {
  id: string
  name: string
  type: string
  params: Record<string, any>
  inputShape?: number[]
  outputShape?: number[]
  position: { x: number; y: number }
}

export interface VisualizationConnection {
  from: string
  to: string
}

export interface ModelVisualization {
  layers: VisualizationLayer[]
  connections: VisualizationConnection[]
  inputShape: number[]
}

export class VisualizationGenerator {
  private nodes: GraphNode[]
  private edges: GraphEdge[]
  private inputNode: GraphNode | null = null

  constructor(graph: GraphIR) {
    this.nodes = graph.nodes
    this.edges = graph.edges
    this.inputNode = this.nodes.find((node) => node.type === "inputNode") || null
  }

  // Input nodes store their dimensions as individual fields (channels/height/
  // width/features), not as a `shape` array — build the array from those.
  private getInputShapeArray(node: GraphNode | null): number[] {
    if (!node) return []
    if (Array.isArray(node.data.shape)) return node.data.shape
    const { channels, depth, height, width, sequence, length, features } = node.data as any
    const dims = [channels, depth, height, width, sequence, length, features].filter(
      (v): v is number => typeof v === "number",
    )
    // Leading batch dimension for display purposes
    return dims.length > 0 ? [1, ...dims] : []
  }

  // Calculate tensor shapes through the network
  private calculateTensorShapes(): Map<string, number[]> {
    const shapes = new Map<string, number[]>()

    // Start with input shape
    const inputShapeArray = this.getInputShapeArray(this.inputNode)
    if (this.inputNode && inputShapeArray.length > 0) {
      shapes.set(this.inputNode.id, inputShapeArray)
    }

    // Topologically sort nodes
    const sortedNodes = this.topologicalSort()

    for (const node of sortedNodes) {
      if (node.type === "inputNode") continue

      // Get input shape from previous layer
      const inputEdges = this.edges.filter((edge) => edge.target === node.id)
      if (inputEdges.length === 0) continue

      const inputShape = shapes.get(inputEdges[0].source)
      if (!inputShape) continue

      // Calculate output shape based on layer type
      const outputShape = this.calculateLayerOutputShape(node, inputShape)
      if (outputShape) {
        shapes.set(node.id, outputShape)
      }
    }

    return shapes
  }

  private calculateLayerOutputShape(node: GraphNode, inputShape: number[]): number[] | null {
    switch (node.type) {
      case "linearNode":
        // Linear layer: [batch, in_features] -> [batch, out_features]
        return [inputShape[0], node.data.out_features || inputShape[1]]

      case "conv2dNode": {
        // Conv2D: [batch, in_channels, height, width] -> [batch, out_channels, new_height, new_width]
        if (inputShape.length !== 4) return inputShape
        const [kernelH, kernelW] = toScalarPair(node.data.kernel_size, 3)
        const [strideH, strideW] = toScalarPair(node.data.stride, 1)
        const [padH, padW] = toScalarPair(node.data.padding, 0)
        const newHeight = Math.floor((inputShape[2] + 2 * padH - kernelH) / strideH + 1)
        const newWidth = Math.floor((inputShape[3] + 2 * padW - kernelW) / strideW + 1)
        return [inputShape[0], node.data.out_channels || inputShape[1], newHeight, newWidth]
      }

      case "maxpool2dNode": {
        // MaxPool2D: reduces spatial dimensions
        if (inputShape.length !== 4) return inputShape
        const [poolKernelH, poolKernelW] = toScalarPair(node.data.kernel_size, 2)
        const [poolStrideH, poolStrideW] = toScalarPair(node.data.stride ?? node.data.kernel_size, 2)
        const pooledHeight = Math.floor((inputShape[2] - poolKernelH) / poolStrideH + 1)
        const pooledWidth = Math.floor((inputShape[3] - poolKernelW) / poolStrideW + 1)
        return [inputShape[0], inputShape[1], pooledHeight, pooledWidth]
      }

      case "flattenNode":
        // Flatten: [batch, ...] -> [batch, flattened_size]
        const startDim = node.data.start_dim ?? 1
        const flattenedSize = inputShape.slice(startDim).reduce((a, b) => a * b, 1)
        return [inputShape[0], flattenedSize]

      case "dropoutNode":
      case "reluNode":
      case "sigmoidNode":
      case "tanhNode":
      case "batchnorm2dNode":
      case "addNode":
      case "multiplyNode":
        // These layers don't change shape
        return inputShape

      case "lstmNode":
        // LSTM: [batch, seq_len, input_size] -> [batch, seq_len, hidden_size]
        if (inputShape.length !== 3) return inputShape
        return [inputShape[0], inputShape[1], node.data.hidden_size || inputShape[2]]

      default:
        return inputShape
    }
  }

  private topologicalSort(): GraphNode[] {
    const inDegree = new Map<string, number>()
    const adjList = new Map<string, string[]>()

    // Initialize
    for (const node of this.nodes) {
      inDegree.set(node.id, 0)
      adjList.set(node.id, [])
    }

    // Build adjacency list and calculate in-degrees
    for (const edge of this.edges) {
      adjList.get(edge.source)?.push(edge.target)
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1)
    }

    // Kahn's algorithm
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

  generateVisualization(): ModelVisualization {
    const shapes = this.calculateTensorShapes()
    const sortedNodes = this.topologicalSort()

    const layers: VisualizationLayer[] = []
    const connections: VisualizationConnection[] = []

    // Generate layers with calculated positions
    let yPosition = 50
    const xPosition = 200

    for (let i = 0; i < sortedNodes.length; i++) {
      const node = sortedNodes[i]
      // Input shape comes from the node's actual predecessor in the graph,
      // not from whichever node happens to precede it in topological order
      const incomingEdge = this.edges.find((edge) => edge.target === node.id)
      const inputShape = incomingEdge ? shapes.get(incomingEdge.source) : undefined
      const outputShape = shapes.get(node.id)

      layers.push({
        id: node.id,
        name: this.getLayerDisplayName(node),
        type: node.type,
        params: node.data,
        inputShape,
        outputShape,
        position: { x: xPosition, y: yPosition },
      })

      yPosition += 120
    }

    // Generate connections
    for (const edge of this.edges) {
      connections.push({
        from: edge.source,
        to: edge.target,
      })
    }

    return {
      layers,
      connections,
      inputShape: this.getInputShapeArray(this.inputNode),
    }
  }

  private getLayerDisplayName(node: GraphNode): string {
    const manifest = PYTORCH_LAYER_MANIFEST[node.type as keyof typeof PYTORCH_LAYER_MANIFEST]
    if (manifest?.className) {
      return manifest.className.replace("nn.", "")
    }
    return node.type.replace("Node", "")
  }

  // Generate SVG visualization
  generateSVG(): string {
    const visualization = this.generateVisualization()
    const { layers, connections } = visualization

    const width = 600
    const height = Math.max(400, layers.length * 120 + 100)

    let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <style>
          .layer-box { fill: #374151; stroke: #6b7280; stroke-width: 2; rx: 8; }
          .input-box { fill: #8b5cf6; stroke: #7c3aed; }
          .linear-box { fill: #06b6d4; stroke: #0891b2; }
          .conv-box { fill: #10b981; stroke: #059669; }
          .activation-box { fill: #f59e0b; stroke: #d97706; }
          .multiply-box { fill: #a855f7; stroke: #9333ea; }
          .layer-text { fill: white; font-family: 'Geist Sans', sans-serif; font-size: 14px; font-weight: 600; }
          .param-text { fill: #d1d5db; font-family: 'Geist Sans', sans-serif; font-size: 11px; }
          .shape-text { fill: #9ca3af; font-family: 'Geist Mono', monospace; font-size: 10px; }
          .connection { stroke: #6b7280; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
        </style>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
          <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280" />
        </marker>
      </defs>
      <rect width="100%" height="100%" fill="#1f2937"/>
    `

    // Draw connections first (so they appear behind layers)
    for (const connection of connections) {
      const fromLayer = layers.find((l) => l.id === connection.from)
      const toLayer = layers.find((l) => l.id === connection.to)

      if (fromLayer && toLayer) {
        const x1 = fromLayer.position.x + 100
        const y1 = fromLayer.position.y + 40
        const x2 = toLayer.position.x + 100
        const y2 = toLayer.position.y

        svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" class="connection" />`
      }
    }

    // Draw layers
    for (const layer of layers) {
      const { x, y } = layer.position
      const boxClass = this.getLayerBoxClass(layer.type)

      // Layer box
      svg += `<rect x="${x}" y="${y}" width="200" height="80" class="layer-box ${boxClass}" />`

      // Layer name
      svg += `<text x="${x + 100}" y="${y + 20}" text-anchor="middle" class="layer-text">${layer.name}</text>`

      // Parameters
      const paramText = this.formatLayerParams(layer)
      if (paramText) {
        svg += `<text x="${x + 100}" y="${y + 35}" text-anchor="middle" class="param-text">${paramText}</text>`
      }

      // Input/Output shapes
      if (layer.inputShape) {
        svg += `<text x="${x + 10}" y="${y + 55}" class="shape-text">In: [${layer.inputShape.join(", ")}]</text>`
      }
      if (layer.outputShape) {
        svg += `<text x="${x + 10}" y="${y + 70}" class="shape-text">Out: [${layer.outputShape.join(", ")}]</text>`
      }
    }

    svg += "</svg>"
    return svg
  }

  private getLayerBoxClass(type: string): string {
    if (type === "inputNode") return "input-box"
    if (type === "linearNode") return "linear-box"
    if (type === "conv2dNode") return "conv-box"
    if (["reluNode", "sigmoidNode", "tanhNode"].includes(type)) return "activation-box"
    if (type === "multiplyNode") return "multiply-box"
    return ""
  }

  private formatLayerParams(layer: VisualizationLayer): string {
    const params = layer.params
    switch (layer.type) {
      case "linearNode":
        return `${params.in_features || "?"} → ${params.out_features || "?"}`
      case "conv2dNode":
        return `${params.in_channels || "?"}→${params.out_channels || "?"}, k=${params.kernel_size || "?"}`
      case "maxpool2dNode":
        return `kernel=${params.kernel_size || "?"}, stride=${params.stride || "?"}`
      case "dropoutNode":
        return `p=${params.p || 0.5}`
      case "lstmNode":
        return `${params.input_size || "?"} → ${params.hidden_size || "?"}`
      case "batchnorm2dNode":
        return `features=${params.num_features || "?"}`
      case "addNode":
      case "multiplyNode":
        return `inputs: ${params.num_inputs || 2}`
      default:
        return ""
    }
  }
}
