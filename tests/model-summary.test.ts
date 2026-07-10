import { describe, it, expect } from "vitest"
import { formatModelSummary } from "../lib/model-summary"
import type { ModelAnalysis } from "../lib/model-analyzer"

const fakeAnalysis: ModelAnalysis = {
  totalParameters: 203530,
  trainableParameters: 203402,
  totalFLOPs: 407060,
  memoryUsageMB: 1.5,
  modelSizeMB: 0.78,
  estimatedInferenceTimeMs: 0.4,
  layers: [
    {
      name: "fc1",
      type: "linearNode",
      parameters: 200960,
      trainableParameters: 200960,
      flops: 401920,
      memoryMB: 0.8,
      inputShape: { features: 784 },
      outputShape: { features: 256 },
    },
    {
      name: "bn1",
      type: "batchnorm1dNode",
      parameters: 1024,
      trainableParameters: 512,
      flops: 512,
      memoryMB: 0.1,
      inputShape: { features: 256 },
      outputShape: { features: 256 },
    },
    {
      name: "head",
      type: "linearNode",
      parameters: 1546,
      trainableParameters: 1546,
      flops: 3092,
      memoryMB: 0.01,
      inputShape: { features: 256 },
      // Missing/dynamic output shape should render safely.
      outputShape: { features: "dynamic" },
    },
  ],
}

describe("formatModelSummary - markdown", () => {
  const out = formatModelSummary(fakeAnalysis, { format: "markdown" })

  it("renders a markdown table header", () => {
    expect(out).toContain("| Layer (type) | Output Shape | Param # |")
    expect(out).toContain("| --- | --- | --- |")
  })

  it("includes each layer row with type:name label", () => {
    expect(out).toContain("fc1 (linearNode)")
    expect(out).toContain("bn1 (batchnorm1dNode)")
    expect(out).toContain("head (linearNode)")
  })

  it("formats param counts with thousands separators", () => {
    expect(out).toContain("200,960")
  })

  it("renders shapes and dynamic dims", () => {
    expect(out).toContain("[256]")
    expect(out).toContain("[dynamic]")
  })

  it("includes the totals footer", () => {
    expect(out).toContain("**Total params:** 203,530")
    expect(out).toContain("**Trainable params:** 203,402")
    expect(out).toContain("**Non-trainable params:** 128")
    expect(out).toContain("**Total FLOPs:** 407,060")
    expect(out).toContain("**Estimated size (MB):** 0.78")
  })
})

describe("formatModelSummary - plain", () => {
  const out = formatModelSummary(fakeAnalysis, { format: "plain" })

  it("includes column headers and layer rows", () => {
    expect(out).toContain("Layer (type)")
    expect(out).toContain("Output Shape")
    expect(out).toContain("Param #")
    expect(out).toContain("fc1 (linearNode)")
    expect(out).toContain("bn1 (batchnorm1dNode)")
    expect(out).toContain("head (linearNode)")
  })

  it("includes the totals footer", () => {
    expect(out).toContain("Total params: 203,530")
    expect(out).toContain("Trainable params: 203,402")
    expect(out).toContain("Non-trainable params: 128")
    expect(out).toContain("Total FLOPs: 407,060")
    expect(out).toContain("Estimated size (MB): 0.78")
  })

  it("uses fixed-width alignment (no markdown pipes for data rows)", () => {
    // Plain format should not contain markdown table pipes.
    expect(out).not.toContain("| Layer (type) |")
  })
})

describe("formatModelSummary - defaults and robustness", () => {
  it("defaults to markdown when no format is given", () => {
    const out = formatModelSummary(fakeAnalysis)
    expect(out).toContain("| Layer (type) | Output Shape | Param # |")
  })

  it("does not crash on an empty analysis", () => {
    const empty: ModelAnalysis = {
      totalParameters: 0,
      trainableParameters: 0,
      totalFLOPs: 0,
      memoryUsageMB: 0,
      modelSizeMB: 0,
      estimatedInferenceTimeMs: 0,
      layers: [],
    }
    const md = formatModelSummary(empty, { format: "markdown" })
    const plain = formatModelSummary(empty, { format: "plain" })
    expect(md).toContain("**Total params:** 0")
    expect(plain).toContain("Total params: 0")
  })

  it("renders — for a missing output shape", () => {
    const analysis: ModelAnalysis = {
      totalParameters: 10,
      trainableParameters: 10,
      totalFLOPs: 0,
      memoryUsageMB: 0,
      modelSizeMB: 0,
      estimatedInferenceTimeMs: 0,
      layers: [
        {
          name: "x",
          type: "reluNode",
          parameters: 0,
          flops: 0,
          memoryMB: 0,
          inputShape: {} as any,
          outputShape: undefined as any,
        },
      ],
    }
    const out = formatModelSummary(analysis, { format: "plain" })
    expect(out).toContain("—")
  })
})
