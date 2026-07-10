import { describe, it, expect } from "vitest"
import { parsePyTorchModel } from "../lib/pytorch-parser"

function nodesByType(result: ReturnType<typeof parsePyTorchModel>) {
  const counts: Record<string, number> = {}
  for (const n of result.nodes) counts[n.type] = (counts[n.type] || 0) + 1
  return counts
}

// Build a lookup from node id -> node for edge-tracing assertions.
function idMap(result: ReturnType<typeof parsePyTorchModel>) {
  return new Map(result.nodes.map((n) => [n.id, n]))
}

describe("parsePyTorchModel", () => {
  it("parses a simple sequential MLP and connects layers in forward order", () => {
    const code = `
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
`
    const result = parsePyTorchModel(code)
    expect(result.errors).toHaveLength(0)
    const counts = nodesByType(result)
    expect(counts.inputNode).toBe(1)
    expect(counts.linearNode).toBe(2)
    expect(counts.reluNode).toBe(1)
    expect(counts.outputNode).toBe(1)
    // in_features/out_features parsed from positional args
    const linears = result.nodes.filter((n) => n.type === "linearNode")
    expect(linears[0].data.in_features).toBe(784)
    expect(linears[0].data.out_features).toBe(256)
  })

  it("reconstructs a residual (skip) connection as an Add node", () => {
    const code = `
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return torch.relu(out)
`
    const result = parsePyTorchModel(code)
    const counts = nodesByType(result)
    expect(counts.addNode).toBe(1)
    expect(counts.conv2dNode).toBe(2)
    expect(counts.batchnorm2dNode).toBe(2)

    // The Add node must have two distinct incoming edges (skip + main path)
    // with ordered target handles.
    const addNode = result.nodes.find((n) => n.type === "addNode")!
    const addEdges = result.edges.filter((e) => e.target === addNode.id)
    expect(addEdges).toHaveLength(2)
    const handles = addEdges.map((e) => e.targetHandle).sort()
    expect(handles).toEqual(["input1", "input2"])

    // One of the Add inputs should trace straight back to the input node
    // (the identity/skip branch).
    const map = idMap(result)
    const inputNode = result.nodes.find((n) => n.type === "inputNode")!
    const addSources = addEdges.map((e) => e.source)
    expect(addSources).toContain(inputNode.id)
  })

  it("reconstructs concatenation (U-Net style skip) as a Concatenate node", () => {
    const code = `
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(3, 16, 3)
        self.up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final = nn.Conv2d(32, 8, 3)
    def forward(self, x):
        skip = self.down(x)
        y = self.up(skip)
        y = torch.cat([y, skip], dim=1)
        y = self.final(y)
        return y
`
    const result = parsePyTorchModel(code)
    const counts = nodesByType(result)
    expect(counts.concatenateNode).toBe(1)
    const cat = result.nodes.find((n) => n.type === "concatenateNode")!
    expect(cat.data.dim).toBe(1)
    const catEdges = result.edges.filter((e) => e.target === cat.id)
    expect(catEdges).toHaveLength(2)
    expect(catEdges.map((e) => e.targetHandle).sort()).toEqual(["input1", "input2"])
  })

  it("expands nn.Sequential into a chain of nodes", () => {
    const code = `
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    def forward(self, x):
        return self.features(x)
`
    const result = parsePyTorchModel(code)
    const counts = nodesByType(result)
    expect(counts.conv2dNode).toBe(1)
    expect(counts.reluNode).toBe(1)
    expect(counts.maxpool2dNode).toBe(1)
    // input -> conv -> relu -> maxpool -> output : 4 edges
    expect(result.edges.length).toBe(4)
  })

  it("handles LSTM tuple unpacking", () => {
    const code = `
class Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)
`
    const result = parsePyTorchModel(code)
    const counts = nodesByType(result)
    expect(counts.lstmNode).toBe(1)
    expect(counts.linearNode).toBe(1)
    // fc must receive the lstm output (tuple's first element)
    const lstm = result.nodes.find((n) => n.type === "lstmNode")!
    const fc = result.nodes.find((n) => n.type === "linearNode")!
    const edge = result.edges.find((e) => e.source === lstm.id && e.target === fc.id)
    expect(edge).toBeTruthy()
  })

  it("inlines a user-defined submodule's forward graph", () => {
    const code = `
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn = nn.BatchNorm2d(cout)
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(3, 16)
        self.block2 = ConvBlock(16, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x)
`
    const result = parsePyTorchModel(code)
    const counts = nodesByType(result)
    // Two inlined blocks => 2 convs, 2 bns, 2 relus, plus the pool.
    expect(counts.conv2dNode).toBe(2)
    expect(counts.batchnorm2dNode).toBe(2)
    expect(counts.reluNode).toBe(2)
    expect(counts.adaptiveavgpool2dNode).toBe(1)
  })

  it("picks the top-level model when several classes are present", () => {
    const code = `
class Sub(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        return self.fc(x)

class TopModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Sub()
        self.head = nn.Linear(10, 2)
    def forward(self, x):
        x = self.a(x)
        return self.head(x)
`
    const result = parsePyTorchModel(code)
    // Sub is inlined into TopModel; head is TopModel's own layer.
    const counts = nodesByType(result)
    expect(counts.linearNode).toBe(2) // Sub.fc (inlined) + head
    expect(result.nodes.some((n) => n.type === "outputNode")).toBe(true)
  })

  it("assigns non-overlapping layout coordinates and marks a branch column", () => {
    const code = `
class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3,3,1)
        self.c2 = nn.Conv2d(3,3,1)
    def forward(self, x):
        a = self.c1(x)
        b = self.c2(x)
        return a + b
`
    const result = parsePyTorchModel(code)
    // The two convs are at the same depth -> different x positions.
    const convs = result.nodes.filter((n) => n.type === "conv2dNode")
    expect(convs).toHaveLength(2)
    expect(convs[0].position.x).not.toBe(convs[1].position.x)
    expect(convs[0].position.y).toBe(convs[1].position.y)
  })

  it("reports an error for input with no model class", () => {
    const result = parsePyTorchModel("def foo():\n    return 1\n")
    expect(result.errors.length).toBeGreaterThan(0)
  })

  it("keeps unsupported layers as pass-through default nodes with a warning", () => {
    const code = `
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        x = self.embed(x)
        return self.fc(x)
`
    const result = parsePyTorchModel(code)
    expect(result.unsupportedModules).toContain("Embedding")
    expect(result.nodes.some((n) => n.type === "defaultNode")).toBe(true)
    // Graph is still connected end-to-end.
    expect(result.nodes.some((n) => n.type === "outputNode")).toBe(true)
  })
})
