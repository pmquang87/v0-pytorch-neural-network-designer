// torchinfo / `summary()`-style text rendering for a ModelAnalysis.
//
// This module is purely presentational: it takes an ALREADY-COMPUTED analysis
// object (produced by `analyzeModel(nodes, edges)` in ./model-analyzer) and
// formats it as either a GitHub markdown table or a fixed-width plain-text
// table. It never recomputes anything.

import type { ModelAnalysis, LayerAnalysis } from "./model-analyzer"
import type { TensorShape } from "./tensor-shape-calculator"

export interface FormatModelSummaryOptions {
  format?: "markdown" | "plain"
}

// --- Formatting helpers -----------------------------------------------------

const DASH = "—"

// Thousands-separated integer, e.g. 1234567 -> "1,234,567".
function withCommas(n: number): string {
  if (typeof n !== "number" || !isFinite(n)) return DASH
  const rounded = Math.round(n)
  return rounded.toLocaleString("en-US")
}

// Render a TensorShape as a torch-style tuple, e.g. "[1, 32, 8, 8]".
// Missing dimensions render nothing; "dynamic" is preserved verbatim.
function formatShape(shape: TensorShape | undefined | null): string {
  if (!shape || typeof shape !== "object") return DASH
  // Preserve a meaningful ordering of the common tensor dimensions.
  const order: (keyof TensorShape)[] = [
    "channels",
    "features",
    "sequence",
    "depth",
    "height",
    "width",
    "length",
  ]
  const dims: (string | number)[] = []
  for (const key of order) {
    const v = shape[key]
    if (v === undefined || v === null) continue
    dims.push(v === "dynamic" ? "dynamic" : v)
  }
  if (dims.length === 0) return DASH
  return `[${dims.join(", ")}]`
}

// "type:name" label, e.g. "Conv2d (conv2dNode:conv-1)". We keep both the raw
// layer type and the user-facing name, torchinfo-style (Layer (type)).
function formatLayerLabel(layer: LayerAnalysis): string {
  const type = layer.type || DASH
  const name = layer.name || DASH
  return `${name} (${type})`
}

function estimatedSizeMB(analysis: ModelAnalysis): number {
  // Prefer the analyzer's own model size estimate; fall back to param bytes.
  if (typeof analysis.modelSizeMB === "number" && isFinite(analysis.modelSizeMB)) {
    return analysis.modelSizeMB
  }
  return (analysis.totalParameters * 4) / (1024 * 1024)
}

// --- Markdown -----------------------------------------------------------------

function formatMarkdown(analysis: ModelAnalysis): string {
  const header = ["Layer (type)", "Output Shape", "Param #"]
  const lines: string[] = []
  lines.push(`| ${header.join(" | ")} |`)
  lines.push(`| ${header.map(() => "---").join(" | ")} |`)

  for (const layer of analysis.layers ?? []) {
    lines.push(
      `| ${formatLayerLabel(layer)} | ${formatShape(layer.outputShape)} | ${withCommas(
        layer.parameters,
      )} |`,
    )
  }

  const footer: string[] = []
  footer.push("")
  footer.push(`**Total params:** ${withCommas(analysis.totalParameters)}`)
  footer.push(`**Trainable params:** ${withCommas(analysis.trainableParameters)}`)
  footer.push(
    `**Non-trainable params:** ${withCommas(
      (analysis.totalParameters ?? 0) - (analysis.trainableParameters ?? 0),
    )}`,
  )
  footer.push(`**Total FLOPs:** ${withCommas(analysis.totalFLOPs)}`)
  footer.push(`**Estimated size (MB):** ${estimatedSizeMB(analysis).toFixed(2)}`)

  return lines.join("\n") + "\n" + footer.join("\n") + "\n"
}

// --- Plain (fixed-width) ------------------------------------------------------

function padCell(text: string, width: number): string {
  if (text.length >= width) return text
  return text + " ".repeat(width - text.length)
}

function formatPlain(analysis: ModelAnalysis): string {
  const headers = ["Layer (type)", "Output Shape", "Param #"]
  const rows: string[][] = [headers]

  for (const layer of analysis.layers ?? []) {
    rows.push([
      formatLayerLabel(layer),
      formatShape(layer.outputShape),
      withCommas(layer.parameters),
    ])
  }

  // Compute column widths from the widest cell in each column.
  const widths = headers.map((_, col) =>
    rows.reduce((max, row) => Math.max(max, (row[col] ?? "").length), 0),
  )

  const renderRow = (row: string[]) =>
    row.map((cell, col) => padCell(cell ?? "", widths[col])).join("  ")

  const totalWidth = widths.reduce((a, b) => a + b, 0) + (widths.length - 1) * 2
  const divider = "=".repeat(totalWidth)
  const thinDivider = "-".repeat(totalWidth)

  const lines: string[] = []
  lines.push(divider)
  lines.push(renderRow(headers))
  lines.push(divider)
  for (let i = 1; i < rows.length; i++) {
    lines.push(renderRow(rows[i]))
  }
  lines.push(divider)
  lines.push(`Total params: ${withCommas(analysis.totalParameters)}`)
  lines.push(`Trainable params: ${withCommas(analysis.trainableParameters)}`)
  lines.push(
    `Non-trainable params: ${withCommas(
      (analysis.totalParameters ?? 0) - (analysis.trainableParameters ?? 0),
    )}`,
  )
  lines.push(thinDivider)
  lines.push(`Total FLOPs: ${withCommas(analysis.totalFLOPs)}`)
  lines.push(`Estimated size (MB): ${estimatedSizeMB(analysis).toFixed(2)}`)
  lines.push(divider)

  return lines.join("\n") + "\n"
}

// --- Public API ---------------------------------------------------------------

/**
 * Render an already-computed ModelAnalysis as a torchinfo-style summary table.
 * `opts.format` selects "markdown" (a GitHub markdown table, default) or
 * "plain" (fixed-width columns). Missing/"dynamic" shapes render safely.
 */
export function formatModelSummary(
  analysis: ModelAnalysis,
  opts: FormatModelSummaryOptions = {},
): string {
  const format = opts.format ?? "markdown"
  const safe: ModelAnalysis = {
    totalParameters: analysis?.totalParameters ?? 0,
    trainableParameters: analysis?.trainableParameters ?? 0,
    totalFLOPs: analysis?.totalFLOPs ?? 0,
    memoryUsageMB: analysis?.memoryUsageMB ?? 0,
    layers: analysis?.layers ?? [],
    modelSizeMB: analysis?.modelSizeMB ?? 0,
    estimatedInferenceTimeMs: analysis?.estimatedInferenceTimeMs ?? 0,
  }
  return format === "plain" ? formatPlain(safe) : formatMarkdown(safe)
}
