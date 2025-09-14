import { type GraphIR, type GraphNode, type GraphEdge, PYTORCH_LAYER_MANIFEST } from "./types"

export class ModelGenerator {
  private nodes: GraphNode[]
  private edges: GraphEdge[]
  private inputNode: GraphNode | null = null

  constructor(graph: GraphIR) {
    this.nodes = graph.nodes
    this.edges = graph.edges
    this.inputNode = this.nodes.find((node) => node.type === "inputNode") || null
  }

  // Validate the graph structure
  validateGraph(): { valid: boolean; error?: string } {
    // Check for input node
    if (!this.inputNode) {
      return { valid: false, error: "Graph must contain an input node" }
    }

    // Check for cycles (simplified check)
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

  // Topologically sort nodes for execution order
  topologicalSort(): GraphNode[] {
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

    // Start with nodes that have no incoming edges
    for (const [nodeId, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(nodeId)
      }
    }

    while (queue.length > 0) {
      const currentId = queue.shift()!
      const currentNode = this.nodes.find((n) => n.id === currentId)!
      result.push(currentNode)

      // Process neighbors
      for (const neighborId of adjList.get(currentId) || []) {
        inDegree.set(neighborId, inDegree.get(neighborId)! - 1)
        if (inDegree.get(neighborId) === 0) {
          queue.push(neighborId)
        }
      }
    }

    return result
  }

  // Generate PyTorch model code
  generateCode(): string {
    const validation = this.validateGraph()
    if (!validation.valid) {
      throw new Error(validation.error)
    }

    const sortedNodes = this.topologicalSort()
    const layerNodes = sortedNodes.filter((node) => node.type !== "inputNode")

    // Generate imports
    const imports = new Set<string>()
    imports.add("import torch")
    imports.add("import torch.nn as nn")

    // Generate class definition
    let code = `import torch
import torch.nn as nn


class GeneratedModel(nn.Module):
    def __init__(self):
        super(GeneratedModel, self).__init__()
        
`

    for (const node of layerNodes) {
      const manifest = PYTORCH_LAYER_MANIFEST[node.type as keyof typeof PYTORCH_LAYER_MANIFEST]
      if (!manifest || !manifest.className) continue // Skip nodes with null className

      if (node.type === "depthwiseconv2dNode") {
        // For depthwise convolution, set groups = in_channels
        const params = this.buildParameterString(node, manifest.params)
        const layerName = this.sanitizeLayerName(node.id)
        const groups = node.data.in_channels || node.data.groups || 1
        code += `        self.${layerName} = ${manifest.className}(${params}, groups=${groups})\n`
      } else if (node.type === "separableconv2dNode") {
        // For separable convolution, create both depthwise and pointwise layers
        const layerName = this.sanitizeLayerName(node.id)
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
        const layerName = this.sanitizeLayerName(node.id)
        code += `        self.${layerName} = ${manifest.className}(${params})\n`
      }
    }

    // Generate forward method
    code += `
    def forward(self, x):
`

    // Generate forward pass logic
    const forwardLines = this.generateForwardPass(sortedNodes)
    for (const line of forwardLines) {
      code += `        ${line}\n`
    }

    code += `        return x
`

    // Add usage example
    if (this.inputNode?.data.shape) {
      const shape = this.inputNode.data.shape
      code += `

# Usage example:
# model = GeneratedModel()
# input_tensor = torch.randn(${JSON.stringify(shape)})
# output = model(input_tensor)
# print(f"Input shape: {input_tensor.shape}")
# print(f"Output shape: {output.shape}")
`
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
          params.push(`${paramName}="${value}"`)
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

    // Input node starts the chain
    if (this.inputNode) {
      nodeOutputs.set(this.inputNode.id, "x")
    }

    for (const node of sortedNodes) {
      if (node.type === "inputNode") continue

      const layerName = this.sanitizeLayerName(node.id)
      const inputEdges = this.edges.filter((edge) => edge.target === node.id)

      if (inputEdges.length === 0) {
        // No input - skip this node
        continue
      }

      if (node.type === "addNode") {
        // Handle addition operation
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
        const dim = node.data.dim || 1 // Default to dim=1 if not specified

        if (inputVars.length === 2) {
          // For U-Net skip connections, crop the second input (skip connection) to match the first input (decoder)
          const decoderVar = inputVars[0]
          const skipVar = inputVars[1]
          const croppedSkipVar = `${skipVar}_cropped`

          lines.push(`# Crop skip connection to match decoder spatial dimensions`)
          lines.push(`decoder_h, decoder_w = ${decoderVar}.shape[2], ${decoderVar}.shape[3]`)
          lines.push(`skip_h, skip_w = ${skipVar}.shape[2], ${skipVar}.shape[3]`)
          lines.push(`if skip_h != decoder_h or skip_w != decoder_w:`)
          lines.push(`    # Center crop the skip connection to match decoder size`)
          lines.push(`    start_h = (skip_h - decoder_h) // 2`)
          lines.push(`    start_w = (skip_w - decoder_w) // 2`)
          lines.push(`    ${croppedSkipVar} = ${skipVar}[:, :, start_h:start_h+decoder_h, start_w:start_w+decoder_w]`)
          lines.push(`else:`)
          lines.push(`    ${croppedSkipVar} = ${skipVar}`)

          lines.push(`${outputVar} = torch.cat([${decoderVar}, ${croppedSkipVar}], dim=${dim})`)
        } else {
          // For other concatenations, use original logic
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
      } else if (inputEdges.length === 1) {
        // Single input - regular layer
        const inputVar = nodeOutputs.get(inputEdges[0].source) || "x"
        const outputVar = `x_${layerName}`
        lines.push(`${outputVar} = self.${layerName}(${inputVar})`)
        nodeOutputs.set(node.id, outputVar)
      } else {
        // Multiple inputs for regular layers - this shouldn't happen for most layers
        const inputVars = inputEdges.map((edge) => nodeOutputs.get(edge.source) || "x")
        const outputVar = `x_${layerName}`
        lines.push(`# Warning: Multiple inputs to regular layer - using first input`)
        lines.push(`${outputVar} = self.${layerName}(${inputVars[0]})`)
        nodeOutputs.set(node.id, outputVar)
      }
    }

    // Find the final output
    const outputNodes = sortedNodes.filter((node) => {
      return !this.edges.some((edge) => edge.source === node.id)
    })

    if (outputNodes.length > 0 && outputNodes[0].type !== "inputNode") {
      const finalOutput = nodeOutputs.get(outputNodes[0].id)
      if (finalOutput && finalOutput !== "x") {
        lines.push(`x = ${finalOutput}`)
      }
    }

    return lines
  }
}
