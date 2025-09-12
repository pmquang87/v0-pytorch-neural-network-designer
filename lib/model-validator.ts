import React from "react"
import { ValidationResult } from "./types"
import type { Node, Edge } from "@xyflow/react"

export class ModelValidator {
  // Cache for validation results to avoid re-validating unchanged models
  private validationCache: Map<string, ValidationResult> = new Map()
  // Validate entire model
  validateModel(nodes: Node[], edges: Edge[]): ValidationResult {
    // Generate a cache key based on a simplified version of the model
    const cacheKey = this.generateCacheKey(nodes, edges)

    // Check if we have a cached result
    if (this.validationCache.has(cacheKey)) {
      console.log("[ModelValidator] Using cached validation result")
      return this.validationCache.get(cacheKey)!
    }

    console.log("[ModelValidator] Validating model with", nodes.length, "nodes and", edges.length, "edges")

    const errors: string[] = []
    const warnings: string[] = []

    // Check for empty model
    if (nodes.length === 0) {
      warnings.push("Model is empty")
      const result = { isValid: true, errors, warnings }
      this.validationCache.set(cacheKey, result)
      return result
    }

    // First, ensure shapes are propagated correctly through the network
    this.propagateShapes(nodes, edges)

    // Check for input/output structure
    const ioErrors = this.checkInputOutputStructure(nodes, edges)
    errors.push(...ioErrors)

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

    // Create and cache the validation result
    const result = {
      isValid: errors.length === 0,
      errors,
      warnings
    }

    // Store in cache
    this.validationCache.set(cacheKey, result)

    return result
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

  // Check for model completeness
  private checkModelCompleteness(nodes: Node[]): string[] {
    const warnings: string[] = []

    // Check if model has at least one input node
    const inputNodes = nodes.filter(node => node.type === "inputNode")
    if (inputNodes.length === 0) {
      warnings.push("Model does not have an input node. Add an input node to define input tensor dimensions.")
    }

    // Check if model has essential components for common neural networks
    const hasActivation = nodes.some(node => 
      node.type === "reluNode" || 
      node.type === "sigmoidNode" || 
      node.type === "tanhNode" || 
      node.type === "leakyreluNode" || 
      node.type === "geluNode" ||
      node.type === "softmaxNode"
    )

    if (!hasActivation && nodes.length > 2) {
      warnings.push("Model does not have any activation functions, which may limit its ability to learn non-linear patterns.")
    }

    return warnings
  }

  // Check input/output structure
  private checkInputOutputStructure(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []

    // Skip if no nodes
    if (nodes.length === 0) return errors

    // Find nodes with no incoming edges (sources)
    const sourceNodeIds = new Set(nodes.map(node => node.id))
    edges.forEach(edge => sourceNodeIds.delete(edge.target))

    // Sources that aren't input nodes
    const nonInputSources = nodes
      .filter(node => sourceNodeIds.has(node.id) && node.type !== "inputNode")

    if (nonInputSources.length > 0 && nodes.length > 1) {
      nonInputSources.forEach(node => {
        errors.push(`Node ${node.id} (${node.type}) has no input connections but is not an input node`)
      })
    }

    // Find nodes with no outgoing edges (sinks)
    const sinkNodeIds = new Set(nodes.map(node => node.id))
    edges.forEach(edge => sinkNodeIds.delete(edge.source))

    // Check if there are too many sinks (possible unconnected outputs)
    if (sinkNodeIds.size > 1 && nodes.length > 2) {
      errors.push(`Found ${sinkNodeIds.size} nodes with no output connections. Your model may have unintended termination points.`)
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

    // First check edges for shape compatibility
    for (const edge of edges) {
      const sourceNode = nodeMap.get(edge.source)
      const targetNode = nodeMap.get(edge.target)

      if (sourceNode && targetNode) {
        const sourceOutputShape = sourceNode.data?.outputShape
        const targetInputShape = targetNode.data?.inputShape

        if (sourceOutputShape && targetInputShape) {
          // Basic shape compatibility check
          if (!this.shapesCompatible(sourceOutputShape, targetInputShape)) {
            const sourceType = sourceNode.type?.replace('Node', '') || 'unknown'
            const targetType = targetNode.type?.replace('Node', '') || 'unknown'
            errors.push(`Shape mismatch between ${sourceType}-${sourceNode.id} output and ${targetType}-${targetNode.id} input`)
          }
        }
      }
    }

    // Then check individual nodes for internal shape consistency
    for (const node of nodes) {
      // For linear layers, check if in_features matches the input shape features
      if (node.type === "linearNode" && node.data) {
        const { in_features, inputShape } = node.data

        if (in_features !== undefined && inputShape?.features !== undefined && 
            inputShape.features !== "dynamic" && in_features !== inputShape.features) {
          errors.push(`Shape mismatch in ${node.id}: in_features (${in_features}) doesn't match input shape features (${inputShape.features})`)
        }
      }

      // For convolutional layers, check if in_channels matches the input shape channels
      if ((node.type === "conv1dNode" || node.type === "conv2dNode" || node.type === "conv3dNode") && node.data) {
        const { in_channels, inputShape } = node.data

        if (in_channels !== undefined && inputShape?.channels !== undefined && 
            inputShape.channels !== "dynamic" && in_channels !== inputShape.channels) {
          errors.push(`Shape mismatch in ${node.id}: in_channels (${in_channels}) doesn't match input shape channels (${inputShape.channels})`)
        }
      }
    }

    return errors
  }

  // Check if two shapes are compatible
  private shapesCompatible(shape1: any, shape2: any): boolean {
    // Handle undefined cases
    if (!shape1 || !shape2) return true;

    // Extract common dimensions that should match
    const commonDimensions = [
      'batch', 'channels', 'height', 'width', 'depth', 'features'
    ];

    // Check each common dimension
    for (const dim of commonDimensions) {
      // Skip if either doesn't have this dimension defined
      if (shape1[dim] === undefined || shape2[dim] === undefined) continue;

      const val1 = shape1[dim];
      const val2 = shape2[dim];

      // Dynamic dimensions are compatible with anything
      if (val1 === 'dynamic' || val2 === 'dynamic') continue;

      // For numerical values, they must match
      if (val1 !== val2) {
        console.log(`[ModelValidator] Shape mismatch: ${dim} dimension ${val1} ≠ ${val2}`);
        return false;
      }
    }

    // Special handling for linear layers: check if features matches in_features
    if (shape1.features && shape2.in_features && 
        shape1.features !== 'dynamic' && shape1.features !== shape2.in_features) {
      console.log(`[ModelValidator] Linear layer shape mismatch: features ${shape1.features} ≠ in_features ${shape2.in_features}`);
      return false;
    }

    // Also check the other direction
    if (shape1.in_features && shape2.features && 
        shape2.features !== 'dynamic' && shape1.in_features !== shape2.features) {
      console.log(`[ModelValidator] Linear layer shape mismatch: in_features ${shape1.in_features} ≠ features ${shape2.features}`);
      return false;
    }

    // Special handling for convolutional layers: check if channels matches in_channels
    if (shape1.channels && shape2.in_channels && 
        shape1.channels !== 'dynamic' && shape1.channels !== shape2.in_channels) {
      console.log(`[ModelValidator] Conv layer shape mismatch: channels ${shape1.channels} ≠ in_channels ${shape2.in_channels}`);
      return false;
    }

    // Also check the other direction
    if (shape1.in_channels && shape2.channels && 
        shape2.channels !== 'dynamic' && shape1.in_channels !== shape2.channels) {
      console.log(`[ModelValidator] Conv layer shape mismatch: in_channels ${shape1.in_channels} ≠ channels ${shape2.channels}`);
      return false;
    }

    return true;
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

  // Generate a cache key for a model
  private generateCacheKey(nodes: Node[], edges: Edge[]): string {
    // Create a simplified representation of the model for caching
    const simplifiedNodes = nodes.map(node => ({
      id: node.id,
      type: node.type,
      data: JSON.stringify(node.data)
    }));

    const simplifiedEdges = edges.map(edge => ({
      source: edge.source,
      target: edge.target
    }));

    return JSON.stringify({ nodes: simplifiedNodes, edges: simplifiedEdges });
  }

  // Propagate shape information through the network
  private propagateShapes(nodes: Node[], edges: Edge[]): void {
    console.log("[ModelValidator] Propagating shapes")
    const nodeMap = new Map(nodes.map(node => [node.id, node]))

    // Find all input nodes to start propagation
    const inputNodes = nodes.filter(node => node.type === "inputNode")
    const visited = new Set<string>()

    // First, ensure all input nodes have proper output shapes based on their parameters
    for (const inputNode of inputNodes) {
      if (!inputNode.data) inputNode.data = {}

      // Set output shape based on input node parameters
      inputNode.data.outputShape = {
        batch: inputNode.data.batch_size || "dynamic",
        features: inputNode.data.features,
        channels: inputNode.data.channels,
        height: inputNode.data.height,
        width: inputNode.data.width
      }
    }

    // Perform topological sort for processing nodes in order
    const topologicalOrder = this.topologicalSort(nodes, edges)

    // Process nodes in topological order to propagate shapes
    for (const nodeId of topologicalOrder) {
      const node = nodeMap.get(nodeId)
      if (!node || visited.has(nodeId)) continue

      // For each node, first get the input shape from incoming edges
      const incomingEdges = edges.filter(edge => edge.target === nodeId)

      if (incomingEdges.length > 0) {
        // Get source node of the first incoming edge (for simplicity, we'll just use the first one)
        const sourceNode = nodeMap.get(incomingEdges[0].source)

        if (sourceNode?.data?.outputShape) {
          // Copy output shape from source to target's input shape
          if (!node.data) node.data = {}
          node.data.inputShape = { ...sourceNode.data.outputShape }

          // Check if in_features matches features from the input shape for linear layers
          if (node.type === "linearNode" && 
              node.data.inputShape.features !== undefined &&
              node.data.in_features !== undefined &&
              node.data.inputShape.features !== "dynamic" &&
              node.data.in_features !== node.data.inputShape.features) {
            console.log(`[ModelValidator] Linear layer ${nodeId} has mismatched in_features: ` +
                        `expected ${node.data.inputShape.features}, got ${node.data.in_features}`)
          }
        }
      }

      // Now compute the output shape based on node type and parameters
      this.computeOutputShape(node)

      visited.add(nodeId)
    }
  }

  // Compute the output shape for a node based on its type and parameters
  private computeOutputShape(node: Node): void {
    if (!node.data) return

    switch (node.type) {
      case "linearNode":
        // For linear layer, output features = out_features
        node.data.outputShape = {
          ...node.data.inputShape,
          features: node.data.out_features
        }
        break

      case "conv2dNode":
        // For conv2d layers, output shape changes based on parameters
        if (node.data.inputShape && node.data.out_channels) {
          const inputHeight = node.data.inputShape.height
          const inputWidth = node.data.inputShape.width

          // Parse parameters with defaults
          const kernelSize = Array.isArray(node.data.kernel_size) ? 
            node.data.kernel_size : [node.data.kernel_size, node.data.kernel_size]

          const stride = Array.isArray(node.data.stride) ? 
            node.data.stride : [node.data.stride || 1, node.data.stride || 1]

          const padding = Array.isArray(node.data.padding) ? 
            node.data.padding : [node.data.padding || 0, node.data.padding || 0]

          // Calculate output dimensions
          let outputHeight = inputHeight
          let outputWidth = inputWidth

          if (inputHeight !== "dynamic" && inputWidth !== "dynamic") {
            outputHeight = Math.floor((inputHeight + 2 * padding[0] - kernelSize[0]) / stride[0] + 1)
            outputWidth = Math.floor((inputWidth + 2 * padding[1] - kernelSize[1]) / stride[1] + 1)
          }

          node.data.outputShape = {
            ...node.data.inputShape,
            channels: node.data.out_channels,
            height: outputHeight,
            width: outputWidth
          }
        }
        break

      case "maxPool2dNode":
      case "avgPool2dNode":
        // For pooling layers, output shape changes based on parameters
        if (node.data.inputShape) {
          const inputHeight = node.data.inputShape.height
          const inputWidth = node.data.inputShape.width

          // Parse parameters with defaults
          const kernelSize = Array.isArray(node.data.kernel_size) ? 
            node.data.kernel_size : [node.data.kernel_size, node.data.kernel_size]

          const stride = Array.isArray(node.data.stride) ? 
            node.data.stride : [node.data.stride || kernelSize[0], node.data.stride || kernelSize[1]]

          const padding = Array.isArray(node.data.padding) ? 
            node.data.padding : [node.data.padding || 0, node.data.padding || 0]

          // Calculate output dimensions
          let outputHeight = inputHeight
          let outputWidth = inputWidth

          if (inputHeight !== "dynamic" && inputWidth !== "dynamic") {
            outputHeight = Math.floor((inputHeight + 2 * padding[0] - kernelSize[0]) / stride[0] + 1)
            outputWidth = Math.floor((inputWidth + 2 * padding[1] - kernelSize[1]) / stride[1] + 1)
          }

          node.data.outputShape = {
            ...node.data.inputShape,
            height: outputHeight,
            width: outputWidth
          }
        }
        break

      case "flattenNode":
        // Flatten layer combines spatial dimensions into features
        if (node.data.inputShape) {
          const { batch, channels, height, width } = node.data.inputShape

          // If all spatial dimensions are defined and numeric, calculate features
          let features = "dynamic"
          if (channels !== undefined && height !== undefined && width !== undefined &&
              channels !== "dynamic" && height !== "dynamic" && width !== "dynamic") {
            features = channels * height * width
          }

          node.data.outputShape = {
            batch,
            features
          }
        }
        break

      // For activation functions and dropout, shape is preserved
      case "reluNode":
      case "leakyReluNode":
      case "sigmoidNode":
      case "tanhNode":
      case "eluNode":
      case "geluNode":
      case "softmaxNode":
      case "dropoutNode":
      case "batchNorm1dNode":
      case "batchNorm2dNode":
      case "layerNormNode":
        if (node.data.inputShape) {
          node.data.outputShape = { ...node.data.inputShape }
        }
        break

      // For nodes without specific handling, preserve shape
      default:
        if (node.data.inputShape && !node.data.outputShape) {
          node.data.outputShape = { ...node.data.inputShape }
        }
    }
  }

  // Perform topological sort of the graph to process nodes in dependency order
  private topologicalSort(nodes: Node[], edges: Edge[]): string[] {
    const result: string[] = []
    const visited = new Set<string>()
    const temporaryMarks = new Set<string>()

    // Build adjacency list
    const adjacencyList = new Map<string, string[]>()

    // Initialize adjacency list for all nodes
    for (const node of nodes) {
      adjacencyList.set(node.id, [])
    }

    // Add edges to adjacency list
    for (const edge of edges) {
      const adjacentNodes = adjacencyList.get(edge.source) || []
      adjacentNodes.push(edge.target)
      adjacencyList.set(edge.source, adjacentNodes)
    }

    // Visit function for DFS
    const visit = (nodeId: string) => {
      // Check for cycles
      if (temporaryMarks.has(nodeId)) {
        // We have a cycle, but we'll just skip it for shape propagation
        return
      }

      // Skip if already visited
      if (visited.has(nodeId)) return

      // Mark node as being processed
      temporaryMarks.add(nodeId)

      // Visit all adjacent nodes
      const adjacentNodes = adjacencyList.get(nodeId) || []
      for (const adjacentId of adjacentNodes) {
        visit(adjacentId)
      }

      // Mark as visited and add to result
      visited.add(nodeId)
      temporaryMarks.delete(nodeId)
      result.unshift(nodeId) // Add to front to get topological order
    }

    // Visit all nodes
    for (const node of nodes) {
      if (!visited.has(node.id)) {
        visit(node.id)
      }
    }

    return result
  }

  // Clear the validation cache
  clearCache(): void {
    this.validationCache.clear();
    console.log("[ModelValidator] Cache cleared");
  }

  // Test shape compatibility for debugging
  testShapeCompatibility(shape1: any, shape2: any): boolean {
    return this.shapesCompatible(shape1, shape2);
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

  const clearValidationCache = React.useCallback(() => {
    validator.clearCache()
  }, [validator])

  return {
    validateModel,
    validateNode,
    validateEdge,
    clearValidationCache
  }
}
