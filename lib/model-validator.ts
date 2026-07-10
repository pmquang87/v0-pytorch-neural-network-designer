import React from "react"
import { ValidationResult } from "./types"
import type { Node, Edge } from "@xyflow/react"
import { validateTensorShapes, formatTensorShape, type TensorShape } from "./tensor-shape-calculator"

// Non-batch dimensions in canonical order. Kept in sync with the calculator's
// own ordering so "last dimension" / "which dim differs" reporting matches the
// shapes the calculator produces.
const CANONICAL_DIM_ORDER: (keyof TensorShape)[] = [
  "channels",
  "depth",
  "height",
  "width",
  "sequence",
  "length",
  "features",
]

// Conv-family node types whose in_channels must equal the incoming tensor's
// channel dimension. Used to produce an actionable in_channels mismatch message.
const CONV_CHANNEL_NODE_TYPES = new Set<string>([
  "conv1dNode",
  "conv2dNode",
  "conv3dNode",
  "depthwiseconv2dNode",
  "separableconv2dNode",
  "convtranspose1dNode",
  "convtranspose2dNode",
  "convtranspose3dNode",
])

function convDisplayName(type: string | undefined): string {
  const map: Record<string, string> = {
    conv1dNode: "Conv1d",
    conv2dNode: "Conv2d",
    conv3dNode: "Conv3d",
    depthwiseconv2dNode: "DepthwiseConv2d",
    separableconv2dNode: "SeparableConv2d",
    convtranspose1dNode: "ConvTranspose1d",
    convtranspose2dNode: "ConvTranspose2d",
    convtranspose3dNode: "ConvTranspose3d",
  }
  return map[type ?? ""] ?? "Conv"
}

function orderedDimNames(shape: TensorShape): (keyof TensorShape)[] {
  return CANONICAL_DIM_ORDER.filter((d) => (shape as any)[d] !== undefined)
}

function fmtDim(v: number | "dynamic"): string {
  return v === "dynamic" ? "?" : String(v)
}

// First dimension where two Add/Multiply inputs disagree under broadcasting
// rules (dims must be equal, or one of them 1 / absent).
function firstBroadcastMismatch(
  shapes: TensorShape[],
): { key: keyof TensorShape; a: number; b: number } | null {
  for (const key of CANONICAL_DIM_ORDER) {
    const vals: number[] = []
    for (const s of shapes) {
      const raw = (s as any)[key]
      const dim = raw === undefined ? 1 : raw
      if (typeof dim === "number" && dim !== 1) vals.push(dim)
    }
    const distinct = Array.from(new Set(vals))
    if (distinct.length > 1) {
      return { key, a: distinct[0], b: distinct[1] }
    }
  }
  return null
}

// First problem for Concatenate: either a differing rank, or a non-concat
// dimension whose sizes disagree. codeDim is the 0-based concat axis.
function firstConcatMismatch(
  shapes: TensorShape[],
  codeDim: number,
):
  | { kind: "rank"; aLen: number; bLen: number }
  | { kind: "dim"; index: number; a: number | "dynamic"; b: number | "dynamic" }
  | null {
  const dimNames = shapes.map(orderedDimNames)
  const len = dimNames[0].length
  for (let i = 1; i < dimNames.length; i++) {
    if (dimNames[i].length !== len) {
      return { kind: "rank", aLen: len, bLen: dimNames[i].length }
    }
  }
  for (let idx = 0; idx < len; idx++) {
    if (idx === codeDim) continue
    let first: number | "dynamic" | undefined
    for (let s = 0; s < shapes.length; s++) {
      const v = (shapes[s] as any)[dimNames[s][idx]] as number | "dynamic" | undefined
      if (v === undefined) continue
      if (first === undefined) {
        first = v
      } else if (first !== v && first !== "dynamic" && v !== "dynamic") {
        return { kind: "dim", index: idx, a: first, b: v }
      }
    }
  }
  return null
}

// Build an actionable message for Add/Multiply/Concatenate incompatibilities,
// naming both shapes and the offending dimension. Falls back to the calculator's
// own error text if the specific offending dim can't be localized.
function describeOpMismatch(
  nodeType: string,
  nodeData: any,
  shapes: TensorShape[],
  fallback: string,
): string {
  const shapeList = shapes.map((s) => formatTensorShape(s)).join(" and ")

  if (nodeType === "concatenateNode") {
    const codeDim = Number(nodeData?.dim ?? 1) - 1
    const detail = firstConcatMismatch(shapes, codeDim)
    let clause = ""
    if (detail?.kind === "rank") {
      clause = `; one input has ${detail.aLen} dimensions and another has ${detail.bLen}`
    } else if (detail?.kind === "dim") {
      clause = `; dimension ${detail.index + 1} differs (${fmtDim(detail.a)} vs ${fmtDim(detail.b)})`
    }
    if (!detail) return fallback
    return `Concatenate error: cannot concatenate ${shapeList} — all non-concatenated dimensions must match${clause}.`
  }

  const name = nodeType === "addNode" ? "Add" : "Multiply"
  const detail = firstBroadcastMismatch(shapes)
  if (!detail) return fallback
  return `${name} error: cannot combine ${shapeList} — all dimensions must match or be broadcastable; dimension '${detail.key}' differs (${detail.a} vs ${detail.b}).`
}

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

    for (const node of nodes) {
      if (node.type === "inputNode") continue

      const inputEdges = edges.filter((edge) => edge.target === node.id)
      let inputShapes: (TensorShape | undefined)[] = []

      if (node.type === "concatenateNode" || node.type === "addNode" || node.type === "multiplyNode") {
        // Consider every connected inputN handle, not just a fixed count
        const handleIndices = inputEdges
          .map((e) => {
            const match = /^input(\d+)$/.exec(e.targetHandle ?? "")
            return match ? Number(match[1]) : 0
          })
          .filter((n) => n > 0)
        const declaredInputs = Number(node.data.num_inputs ?? node.data.inputs ?? 2)
        const numInputs = Math.max(declaredInputs, ...handleIndices, 2)
        for (let i = 1; i <= numInputs; i++) {
          const handleId = `input${i}`
          const edge = inputEdges.find((e) => e.targetHandle === handleId)
          if (edge) {
            const sourceNode = nodeMap.get(edge.source)
            const sourceShape = (sourceNode?.data as any)?.outputShape as TensorShape | undefined
            inputShapes.push(sourceShape)
          } else {
            inputShapes.push(undefined)
          }
        }
      } else if (node.type === "multiheadattentionNode") {
        const queryEdge = inputEdges.find((e) => e.targetHandle === "query")
        if (queryEdge) {
          const sourceNode = nodeMap.get(queryEdge.source)
          const sourceShape = (sourceNode?.data as any)?.outputShape as TensorShape | undefined
          inputShapes.push(sourceShape)
        }
      } else {
        if (inputEdges.length === 0) continue
        inputShapes = inputEdges.map((edge) => {
          const sourceNode = nodeMap.get(edge.source)
          const sourceShape = (sourceNode?.data as any)?.outputShape as TensorShape | undefined
          return sourceShape
        })
      }

      const connectedShapes = inputShapes.filter((s): s is TensorShape => !!s)


      if (node.type === "addNode" || node.type === "multiplyNode" || node.type === "concatenateNode") {
        if (connectedShapes.length < 2) {
          continue
        }
      } else {
        if (connectedShapes.length === 0) {
          continue
        }
      }

      const label = (node.data.label as string) || node.id

      // Linear: nn.Linear only transforms the LAST dimension. State expected vs
      // actual and offer the concrete fixes (Flatten / adjust in_features /
      // select a single position). This is the BERT/T5 class of bug.
      if (node.type === "linearNode") {
        const inFeatures = Number(node.data?.in_features)
        const incoming = inputShapes.find(
          (s): s is TensorShape => !!s && Object.keys(s).length > 0,
        )
        if (Number.isFinite(inFeatures) && inFeatures > 0 && incoming) {
          const dims = orderedDimNames(incoming)
          const last = dims.length ? (incoming as any)[dims[dims.length - 1]] : undefined
          if (typeof last === "number" && last !== inFeatures) {
            errors.push(
              `Node '${label}' (linearNode): Shape mismatch — Linear expects in_features=${inFeatures} but receives a tensor whose last dimension is ${last}. Insert a Flatten node before this Linear, set in_features=${last}, or select a single token/position.`,
            )
          }
        }
        continue
      }

      // Conv family: in_channels must equal the incoming tensor's channel count.
      if (CONV_CHANNEL_NODE_TYPES.has(node.type || "")) {
        const inChannels = Number(node.data?.in_channels)
        const incoming = inputShapes.find(
          (s): s is TensorShape => !!s && Object.keys(s).length > 0,
        )
        const incomingChannels = incoming?.channels
        if (
          Number.isFinite(inChannels) &&
          inChannels > 0 &&
          typeof incomingChannels === "number" &&
          incomingChannels !== inChannels
        ) {
          const convName = convDisplayName(node.type)
          errors.push(
            `Node '${label}' (${node.type}): Shape mismatch — ${convName} expects in_channels=${inChannels} but the incoming tensor has ${incomingChannels} channels. Set in_channels=${incomingChannels} to match the previous layer's output.`,
          )
        }
        continue
      }

      const result = validateTensorShapes(node.type || "", inputShapes, node.data)
      if (result && !result.isValid && result.error) {
        if (
          node.type === "addNode" ||
          node.type === "multiplyNode" ||
          node.type === "concatenateNode"
        ) {
          errors.push(
            `Node '${label}' (${node.type}): ${describeOpMismatch(node.type, node.data, connectedShapes, result.error)}`,
          )
        } else {
          errors.push(`Node '${label}' (${node.type}): ${result.error}`)
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

      const label = node.data.label || node.id

      switch (nodeType) {
        case "linearNode": {
          const missing: string[] = []
          if (!nodeData?.in_features)
            missing.push("in_features — set it to the size of the incoming feature dimension (e.g. 784)")
          if (!nodeData?.out_features)
            missing.push("out_features — set it to the desired output size (e.g. 128)")
          if (missing.length) {
            errors.push(`Linear node ${label} is missing ${missing.join("; and ")}.`)
          }
          break
        }
        case "conv2dNode": {
          const missing: string[] = []
          if (!nodeData?.in_channels)
            missing.push("in_channels — set it to the number of input channels from the previous layer (e.g. 3)")
          if (!nodeData?.out_channels)
            missing.push("out_channels — set it to the number of output feature maps (e.g. 64)")
          if (!nodeData?.kernel_size)
            missing.push("kernel_size — set the convolution window size (e.g. 3)")
          if (missing.length) {
            errors.push(`Conv2d node ${label} is missing ${missing.join("; and ")}.`)
          }
          break
        }
        case "dropoutNode":
          {
            const p = Number(nodeData?.p)
            if (!Number.isFinite(p) || p < 0 || p > 1) {
              errors.push(`Dropout node ${node.data.label || node.id} has invalid probability value (should be between 0 and 1)`)
            }
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
        const inFeatures = Number(node.data?.in_features ?? 0)
        const outFeatures = Number(node.data?.out_features ?? 0)
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

  // Test shape compatibility between two shapes (for testing purposes)
  testShapeCompatibility(shape1: any, shape2: any): boolean {
    // Handle dynamic batch sizes
    if (shape1.batch !== shape2.batch) {
      if (shape1.batch !== 'dynamic' && shape2.batch !== 'dynamic') {
        // Different non-dynamic batch sizes are still compatible for shape testing
      }
    }

    // Check features match
    if (shape1.features !== undefined && shape2.features !== undefined) {
      if (shape1.features !== shape2.features &&
        shape1.features !== 'dynamic' && shape2.features !== 'dynamic') {
        return false;
      }
    }

    // Check if shape2 has in_features (for linear layers)
    if (shape2.in_features !== undefined && shape1.features !== undefined) {
      if (shape1.features !== shape2.in_features &&
        shape1.features !== 'dynamic' && shape2.in_features !== 'dynamic') {
        return false;
      }
    }

    // Check channels match
    if (shape1.channels !== undefined && shape2.channels !== undefined) {
      if (shape1.channels !== shape2.channels &&
        shape1.channels !== 'dynamic' && shape2.channels !== 'dynamic') {
        return false;
      }
    }

    // Check if shape2 has in_channels (for conv layers)
    if (shape2.in_channels !== undefined && shape1.channels !== undefined) {
      if (shape1.channels !== shape2.in_channels &&
        shape1.channels !== 'dynamic' && shape2.in_channels !== 'dynamic') {
        return false;
      }
    }

    return true;
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
