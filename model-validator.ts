import React from "react"
import { ValidationResult, GraphNode, GraphEdge } from "./types"
import type { Node, Edge } from "@xyflow/react"

export class ModelValidator {
  // Cache for validation results to avoid re-validating unchanged models
  private validationCache: Map<string, ValidationResult> = new Map()
  // Validate entire model
  validateModel(nodes: Node[], edges: Edge[]): ValidationResult {
