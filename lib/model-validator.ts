import React from "react"
import { ValidationResult } from "./types"
import type { Node, Edge } from "@xyflow/react"
import { validateTensorShapes, type TensorShape } from "./tensor-shape-calculator"

export class ModelValidator {
  // Validate entire model
  validateModel(nodes: Node[], edges: Edge[]): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    // Check for empty model
    if (nodes.length === 0) {
      warnings.push("Model is empty")
    }

    // Check for cycles
    const cycleErrors = this.checkForCycles(nodes, edges)
    errors.push(...cycleErrors)

    // Check for disconnected nodes
    const disconnectedErrors = this.checkDisconnectedNodes(nodes, edges)
    warnings.push(...disconnectedErrors)

    // Check for multiple input nodes
    const inputNodeWarnings = this.checkMultipleInputNodes(nodes)
    warnings.push(...inputNodeWarnings)

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
      errors: [...new Set(errors)], // Remove duplicates
      warnings: [...new Set(warnings)],
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

      const outgoingEdges = edges.filter((edge) => edge.source === nodeId)
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
        errors.push(`Cycle detected in the graph starting from node: ${node.data.label || node.id}`)
      }
    }

    return errors
  }

  // Check for disconnected nodes
  private checkDisconnectedNodes(nodes: Node[], edges: Edge[]): string[] {
    if (nodes.length <= 1) return []
    const warnings: string[] = []
    const connectedNodes = new Set<string>()

    edges.forEach((edge) => {
      connectedNodes.add(edge.source)
      connectedNodes.add(edge.target)
    })

    const disconnected = nodes.filter((node) => !connectedNodes.has(node.id))

    if (disconnected.length > 0) {
      warnings.push(
        `Found ${disconnected.length} disconnected node(s): ${disconnected.map((n) => n.data.label || n.id).join(", ")}`,
      )
    }

    return warnings
  }

  // Check for multiple input nodes
  private checkMultipleInputNodes(nodes: Node[]): string[] {
    const warnings: string[] = []
    const inputNodes = nodes.filter((node) => node.type === "inputNode")

    if (inputNodes.length > 1) {
      warnings.push(
        `Multiple input nodes found (${inputNodes.length}).`,
      )
    }

    return warnings
  }

  // Check for invalid connections
  private checkInvalidConnections(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []
    const nodeIds = new Set(nodes.map((node) => node.id))

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

  // Check for shape mismatches across the model
  private checkShapeMismatches(nodes: Node[], edges: Edge[]): string[] {
    const errors: string[] = []
    const nodeMap = new Map(nodes.map((node) => [node.id, node]))

    // General validation using the centralized `validateTensorShapes` function
    for (const node of nodes) {
      if (node.type === "inputNode") continue

      const inputEdges = edges.filter((edge) => edge.target === node.id)
      if (inputEdges.length === 0) continue

      let inputShapes: (TensorShape | undefined)[] = []

      if (node.type === "concatenateNode" || node.type === "addNode") {
        const numInputs = node.data.num_inputs || node.data.inputs || 2
        for (let i = 1; i <= numInputs; i++) {
          const handleId = `input${i}`
          const edge = inputEdges.find((e) => e.targetHandle === handleId)
          if (edge) {
            const sourceNode = nodeMap.get(edge.source)
            inputShapes.push(sourceNode?.data.outputShape)
          } else {
            inputShapes.push(undefined)
          }
        }
      } else if (node.type === "multiheadattentionNode") {
        const queryEdge = inputEdges.find((e) => e.targetHandle === "query")
        if (queryEdge) {
          const sourceNode = nodeMap.get(queryEdge.source)
          inputShapes.push(sourceNode?.data.outputShape)
        }
      } else {
        inputShapes = inputEdges.map((edge) => {
          const sourceNode = nodeMap.get(edge.source)
          return sourceNode?.data.outputShape
        })
      }

      const cleanInputShapes = inputShapes.filter((s): s is TensorShape => !!s)

      if (cleanInputShapes.length > 0) {
        if ((node.type === "concatenateNode" || node.type === "addNode") && cleanInputShapes.length < 2) {
          continue
        }

        const result = validateTensorShapes(node.type || "", cleanInputShapes, node.data)
        if (result && !result.isValid && result.error) {
          errors.push(`Node '${node.data.label || node.id}' (${node.type}): ${result.error}`)
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
            errors.push(`Linear node ${node.data.label || node.id} is missing required parameters: in_features, out_features`)
          }
          break
        case "conv2dNode":
          if (!nodeData?.in_channels || !nodeData?.out_channels || !nodeData?.kernel_size) {
            errors.push(
              `Conv2D node ${node.data.label || node.id} is missing required parameters: in_channels, out_channels, kernel_size`,
            )
          }
          break
        case "dropoutNode":
          if (nodeData?.p === undefined || nodeData?.p < 0 || nodeData?.p > 1) {
            errors.push(`Dropout node ${node.data.label || node.id} has invalid probability value (should be between 0 and 1)`)
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

    if (nodes.length > 100) {
      warnings.push(`Model has ${nodes.length} nodes, which might impact performance`)
    }

    if (edges.length > 200) {
      warnings.push(`Model has ${edges.length} connections, which might impact performance`)
    }

    const largeLinearLayers = nodes.filter((node) => {
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

    return warnings
  }

  // Validate individual node
  validateNode(node: Node): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    if (!node.data) {
      errors.push(`Node ${node.id} is missing data`)
    }

    switch (node.type) {
      case "linearNode":
        if (!node.data?.in_features || !node.data?.out_features) {
          errors.push(`Linear node ${node.id} is missing in_features or out_features`)
        }
        break
      // Add more validations as needed
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    }
  }

  // Validate edge
  validateEdge(edge: Edge, nodes: Node[]): ValidationResult {
    const errors: string[] = []
    const warnings: string[] = []

    const sourceNode = nodes.find((n) => n.id === edge.source)
    const targetNode = nodes.find((n) => n.id === edge.target)

    if (!sourceNode || !targetNode) {
      errors.push(`Edge references a non-existent node`)
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
    }
  }
}

// React hook for model validation
export function useModelValidation() {
  const [validator] = React.useState(() => new ModelValidator())

  const validateModel = React.useCallback(
    (nodes: Node[], edges: Edge[]) => {
      return validator.validateModel(nodes, edges)
    },
    [validator],
  )

  const validateNode = React.useCallback(
    (node: Node) => {
      return validator.validateNode(node)
    },
    [validator],
  )

  const validateEdge = React.useCallback(
    (edge: Edge, nodes: Node[]) => {
      return validator.validateEdge(edge, nodes)
    },
    [validator],
  )

  return {
    validateModel,
    validateNode,
    validateEdge,
  }
}
