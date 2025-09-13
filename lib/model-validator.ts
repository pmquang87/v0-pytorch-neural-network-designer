import React from "react"
import { ValidationResult } from "./types"
import type { Node, Edge } from "@xyflow/react"

export class ModelValidator {
  // Validate entire model
  validateModel(nodes: Node[], edges: Edge[]): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    // Check for empty model
    if (nodes.length === 0) {
      warnings.push("Model is empty")
      return { isValid: true, errors, warnings }
    }

    // Check for cycles
    const cycleErrors = this.checkForCycles(nodes, edges)
    errors.push(...cycleErrors)

    // Check for disconnected nodes
    const disconnectedErrors = this.checkDisconnectedNodes(nodes, edges)
    warnings.push(...disconnectedErrors)

    // Check for multiple input nodes
    const inputNodeErrors = this.checkMultipleInputNodes(nodes)
    errors.push(...inputNodeErrors)

    // Check for invalid connections
    const connectionErrors = this.checkInvalidConnections(nodes, edges)
    errors.push(...connectionErrors)

    // Check for shape mismatches
    const shapeErrors = this.checkShapeMismatches(nodes, edges)
    errors.push(...shapeErrors)

    // Check for missing required parameters
    const parameterErrors = this.checkRequiredParameters(nodes)
    errors.push(...parameterErrors)

    // Check for performance issues
    const performanceWarnings = this.checkPerformanceIssues(nodes, edges)
    warnings.push(...performanceWarnings)

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    }
  }

  // Check for cycles in the graph
  private checkForCycles(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []
    const visited = new Set<string>()
    const recursionStack = new Set<string>()

    const hasCycle = (nodeId: string): boolean => {
      if (recursionStack.has(nodeId)) {
        return true
      }
      if (visited.has(nodeId)) {
        return false
      }

      visited.add(nodeId)
      recursionStack.add(nodeId)

      const outgoingEdges = edges.filter(edge => edge.source === nodeId)
      for (const edge of outgoingEdges) {
        if (hasCycle(edge.target)) {
          return true
        }
      }

      recursionStack.delete(nodeId)
      return false
    }

    for (const node of nodes) {
      if (!visited.has(node.id) && hasCycle(node.id)) {
        errors.push(`Cycle detected in the graph starting from node: ${node.id}`)
      }
    }

    return errors
  }

  // Check for disconnected nodes
  private checkDisconnectedNodes(nodes: Node[], edges: Edge[]): string[] {
    const warnings: string[] = []
    const connectedNodes = new Set<string>()

    // Add all nodes that have connections
    edges.forEach(edge => {
      connectedNodes.add(edge.source)
      connectedNodes.add(edge.target)
    })

    // Find disconnected nodes
    const disconnectedNodes = nodes.filter(node => !connectedNodes.has(node.id))
    
    if (disconnectedNodes.length > 0) {
      warnings.push(`Found ${disconnectedNodes.length} disconnected node(s): ${disconnectedNodes.map(n => n.id).join(", ")}`)
    }

    return warnings
  }

  // Check for multiple input nodes
  private checkMultipleInputNodes(nodes: Node[]): string[] {
    const errors: string[] = []
    const inputNodes = nodes.filter(node => node.type === "inputNode")
    
    if (inputNodes.length > 1) {
      errors.push(`Multiple input nodes found (${inputNodes.length}). Consider using only one input node or connecting them properly.`)
    }

    return errors
  }

  // Check for invalid connections
  private checkInvalidConnections(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []
    const nodeIds = new Set(nodes.map(node => node.id))

    for (const edge of edges) {
      if (!nodeIds.has(edge.source)) {
        errors.push(`Edge references non-existent source node: ${edge.source}`)
      }
      if (!nodeIds.has(edge.target)) {
        errors.push(`Edge references non-existent target node: ${edge.target}`)
      }
    }

    return errors
  }

  // Check for shape mismatches
  private checkShapeMismatches(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []
    const nodeMap = new Map(nodes.map(node => [node.id, node]))

    for (const edge of edges) {
      const sourceNode = nodeMap.get(edge.source)
      const targetNode = nodeMap.get(edge.target)

      if (sourceNode && targetNode) {
        const sourceOutputShape = sourceNode.data?.outputShape

        if (sourceOutputShape) {
            // Rule for LinearNode: requires a 2D input (batch, features)
            if (targetNode.type === 'linearNode') {
                const isFuzzy2D = sourceOutputShape.features !== undefined;
                const is4D = sourceOutputShape.channels !== undefined || sourceOutputShape.height !== undefined || sourceOutputShape.width !== undefined;

                if (is4D && !isFuzzy2D) { // Clearly a 4D image shape, not flattened
                    errors.push(`Shape mismatch: Node ${targetNode.id} (${targetNode.data.label || 'Linear'}) expects a 2D input (e.g., from a Flatten layer), but received a 4D image-like input from ${sourceNode.id}.`);
                } else if (sourceOutputShape.features && targetNode.data?.in_features && sourceOutputShape.features !== targetNode.data?.in_features) {
                    errors.push(`Shape mismatch: Node ${targetNode.id} (${targetNode.data.label || 'Linear'}) expects in_features=${targetNode.data?.in_features}, but received ${sourceOutputShape.features} from ${sourceNode.id}.`);
                }
            }

            // Rule for Conv2DNode: requires a 4D input (batch, channels, height, width)
            if (targetNode.type === 'conv2dNode') {
                const is4D = sourceOutputShape.channels !== undefined && sourceOutputShape.height !== undefined && sourceOutputShape.width !== undefined;
                if (!is4D) {
                    errors.push(`Shape mismatch: Node ${targetNode.id} (${targetNode.data.label || 'Conv2D'}) expects a 4D image input, but received a different shape from ${sourceNode.id}.`);
                } else if (sourceOutputShape.channels && targetNode.data?.in_channels && sourceOutputShape.channels !== targetNode.data?.in_channels) {
                    errors.push(`Shape mismatch: Node ${targetNode.id} (${targetNode.data.label || 'Conv2D'}) expects in_channels=${targetNode.data?.in_channels}, but received ${sourceOutputShape.channels} from ${sourceNode.id}.`);
                }
            }
        }
      }
    }

    return errors
  }

  // Check for missing required parameters
  private checkRequiredParameters(nodes: Node[]): string[] {
    const errors: string[] = []

    for (const node of nodes) {
      const nodeType = node.type
      const nodeData = node.data

      switch (nodeType) {
        case "linearNode":
          if (!nodeData?.in_features || !nodeData?.out_features) {
            errors.push(`Linear node ${node.id} is missing required parameters: in_features, out_features`)
          }
          break
        case "conv2dNode":
          if (!nodeData?.in_channels || !nodeData?.out_channels || !nodeData?.kernel_size) {
            errors.push(`Conv2D node ${node.id} is missing required parameters: in_channels, out_channels, kernel_size`)
          }
          break
        case "dropoutNode":
          if (nodeData?.p === undefined || nodeData?.p < 0 || nodeData?.p > 1) {
            errors.push(`Dropout node ${node.id} has invalid probability value (should be between 0 and 1)`)
          }
          break
        // Add more node type validations as needed
      }
    }

    return errors
  }

  // Check for performance issues
  private checkPerformanceIssues(nodes: Node[], edges: Edge[]): string[] {
    const warnings: string[] = []

    // Check for too many nodes
    if (nodes.length > 100) {
      warnings.push(`Model has ${nodes.length} nodes, which might impact performance`)
    }

    // Check for too many edges
    if (edges.length > 200) {
      warnings.push(`Model has ${edges.length} connections, which might impact performance`)
    }

    // Check for very large linear layers
    const largeLinearLayers = nodes.filter(node => {
      if (node.type === "linearNode") {
        const inFeatures = node.data?.in_features || 0
        const outFeatures = node.data?.out_features || 0
        return inFeatures > 10000 || outFeatures > 10000
      }
      return false
    })

    if (largeLinearLayers.length > 0) {
      warnings.push(`Found ${largeLinearLayers.length} large linear layer(s) that might impact performance`)
    }

    // Check for very large convolutional layers
    const largeConvLayers = nodes.filter(node => {
      if (node.type === "conv2dNode") {
        const inChannels = node.data?.in_channels || 0
        const outChannels = node.data?.out_channels || 0
        return inChannels > 1000 || outChannels > 1000
      }
      return false
    })

    if (largeConvLayers.length > 0) {
      warnings.push(`Found ${largeConvLayers.length} large convolutional layer(s) that might impact performance`)
    }

    return warnings
  }

  // Validate individual node
  validateNode(node: Node): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    // Check if node has required data
    if (!node.data) {
      errors.push(`Node ${node.id} is missing data`)
    }

    // Check node type specific validations
    switch (node.type) {
      case "inputNode":
        if (!node.data?.batch_size) {
          errors.push(`Input node ${node.id} is missing batch_size`)
        }
        break
      case "linearNode":
        if (!node.data?.in_features || !node.data?.out_features) {
          errors.push(`Linear node ${node.id} is missing in_features or out_features`)
        }
        break
      case "conv2dNode":
        if (!node.data?.in_channels || !node.data?.out_channels || !node.data?.kernel_size) {
          errors.push(`Conv2D node ${node.id} is missing required parameters`)
        }
        break
      // Add more validations as needed
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    }
  }

  // Validate edge
  validateEdge(edge: Edge, nodes: Node[]): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    const sourceNode = nodes.find(n => n.id === edge.source)
    const targetNode = nodes.find(n => n.id === edge.target)

    if (!sourceNode) {
      errors.push(`Edge references non-existent source node: ${edge.source}`)
    }
    if (!targetNode) {
      errors.push(`Edge references non-existent target node: ${edge.target}`)
    }

    if (sourceNode && targetNode) {
      // Check if connection makes sense
      if (sourceNode.type === "inputNode" && targetNode.type === "inputNode") {
        warnings.push(`Connection between two input nodes might not be intended`)
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    }
  }
}

// React hook for model validation
export function useModelValidation() {
  const [validator] = React.useState(() => new ModelValidator())

  const validateModel = React.useCallback((nodes: Node[], edges: Edge[]) => {
    return validator.validateModel(nodes, edges)
  }, [validator])

  const validateNode = React.useCallback((node: Node) => {
    return validator.validateNode(node)
  }, [validator])

  const validateEdge = React.useCallback((edge: Edge, nodes: Node[]) => {
    return validator.validateEdge(edge, nodes)
  }, [validator])

  return {
    validateModel,
    validateNode,
    validateEdge
  }
}
