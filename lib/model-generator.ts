
import { type GraphIR, type GraphNode, type GraphEdge, PYTORCH_LAYER_MANIFEST } from "./types";

// Node types handled inline in the forward pass rather than as nn.Module layers
const FUNCTIONAL_NODE_TYPES = new Set([
  "reshapeNode",
  "addNode",
  "multiplyNode",
  "concatenateNode",
  "transposeNode",
]);

// Maps the MoE node's activation string to the torch.nn class used inside experts.
const MOE_ACTIVATIONS: Record<string, string> = {
  gelu: "nn.GELU",
  silu: "nn.SiLU",
  relu: "nn.ReLU",
};

// Reusable helper module emitted once when a graph contains a Mixture-of-Experts
// node. Implements token-level top-k routing (Mixtral / DeepSeek style): a linear
// gate scores every expert, the top-k are selected and renormalized, and each
// token is processed only by its selected experts before a weighted recombination.
const MOE_HELPER_CLASS = `class MixtureOfExperts(nn.Module):
    """Sparse Mixture-of-Experts feed-forward block with top-k token routing.

    Each token is routed to \`top_k\` of \`num_experts\` expert MLPs by a learned
    gating network; the selected experts' outputs are combined with the
    (renormalized) softmax gate weights. Shape-preserving: (..., d_model) in and out.
    """

    def __init__(self, d_model, d_ff, num_experts, top_k=2, activation=nn.GELU):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        input_shape = x.shape
        x = x.reshape(-1, input_shape[-1])                      # (tokens, d_model)
        routing_weights = torch.softmax(self.gate(x), dim=-1)   # (tokens, num_experts)
        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            token_idx, slot_idx = torch.where(topk_idx == e)
            if token_idx.numel() == 0:
                continue
            weight = topk_weights[token_idx, slot_idx].unsqueeze(-1)
            out[token_idx] += weight * self.experts[e](x[token_idx])
        return out.reshape(input_shape)`;

export class ModelGenerator {
  private readonly nodes: GraphNode[];
  private readonly edges: GraphEdge[];
  private readonly inputNodes: GraphNode[] = [];
  private usesMoE = false;

  constructor(graph: GraphIR) {
    this.nodes = graph.nodes;
    this.edges = graph.edges;
    this.inputNodes = this.nodes.filter((node) => node.type === "inputNode");
  }

  validateGraph(): { valid: boolean; error?: string } {
    if (this.inputNodes.length === 0) {
      return { valid: false, error: "Graph must contain an input node" };
    }

    const visited = new Set<string>();
    const recursionStack = new Set<string>();

    const hasCycle = (nodeId: string): boolean => {
      if (recursionStack.has(nodeId)) return true;
      if (visited.has(nodeId)) return false;

      visited.add(nodeId);
      recursionStack.add(nodeId);

      const outgoingEdges = this.edges.filter((edge) => edge.source === nodeId);
      for (const edge of outgoingEdges) {
        if (hasCycle(edge.target)) return true;
      }

      recursionStack.delete(nodeId);
      return false;
    };

    for (const node of this.nodes) {
      if (hasCycle(node.id)) {
        return { valid: false, error: "Graph contains cycles" };
      }
    }

    return { valid: true };
  }

  private buildGraphStructures(): {
    adjList: Map<string, string[]>;
    inDegree: Map<string, number>;
  } {
    const inDegree = new Map<string, number>();
    const adjList = new Map<string, string[]>();

    for (const node of this.nodes) {
      inDegree.set(node.id, 0);
      adjList.set(node.id, []);
    }

    for (const edge of this.edges) {
      adjList.get(edge.source)?.push(edge.target);
      inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1);
    }
    return { adjList, inDegree };
  }

  topologicalSort(): GraphNode[] {
    const { adjList, inDegree } = this.buildGraphStructures();

    const queue: string[] = [];
    const result: GraphNode[] = [];

    for (const [nodeId, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(nodeId);
      }
    }

    while (queue.length > 0) {
      const currentId = queue.shift()!;
      const currentNode = this.nodes.find((n) => n.id === currentId)!;
      result.push(currentNode);

      for (const neighborId of adjList.get(currentId) || []) {
        inDegree.set(neighborId, inDegree.get(neighborId)! - 1);
        if (inDegree.get(neighborId) === 0) {
          queue.push(neighborId);
        }
      }
    }

    return result;
  }

  generateCode(): string {
    const validation = this.validateGraph();
    if (!validation.valid) {
      throw new Error(validation.error);
    }

    const sortedNodes = this.topologicalSort();
    const layerNodes = sortedNodes.filter(
      (node) => node.type !== "inputNode" && node.type !== "outputNode"
    );

    let code = `import torch\nimport torch.nn as nn\n`;

    let classDefinition = `\n\nclass GeneratedModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n`;

    const initLines: string[] = [];
    const definedLayers = new Set<string>();
    for (const node of layerNodes) {
        const manifest = PYTORCH_LAYER_MANIFEST[node.type as keyof typeof PYTORCH_LAYER_MANIFEST];
        const layerName = this.sanitizeArgName(node.id);

        if (manifest && manifest.className) {
            const params = this.buildParameterString(node, manifest.params);
            initLines.push(`        self.${layerName} = ${manifest.className}(${params})`);
            definedLayers.add(node.id);
        } else if (node.type === "separableconv2dNode") {
            const d = node.data as any;
            const inCh = d.in_channels ?? 1;
            const outCh = d.out_channels ?? 1;
            const k = this.toPythonLiteral(d.kernel_size ?? 3);
            const stride = this.toPythonLiteral(d.stride ?? 1);
            const padding = this.toPythonLiteral(d.padding ?? 0);
            initLines.push(
                `        self.${layerName} = nn.Sequential(\n` +
                `            nn.Conv2d(${inCh}, ${inCh}, kernel_size=${k}, stride=${stride}, padding=${padding}, groups=${inCh}),\n` +
                `            nn.Conv2d(${inCh}, ${outCh}, kernel_size=1),\n` +
                `        )`
            );
            definedLayers.add(node.id);
        } else if (node.type === "parameterNode") {
            const shape = Array.isArray((node.data as any).shape) ? (node.data as any).shape : [1];
            initLines.push(`        self.${layerName} = nn.Parameter(torch.randn(${shape.join(", ")}))`);
            definedLayers.add(node.id);
        } else if (node.type === "moeNode") {
            const d = node.data as any;
            const dModel = d.d_model ?? 512;
            const dFf = d.d_ff ?? dModel * 4;
            const numExperts = d.num_experts ?? 8;
            const topK = d.top_k ?? 2;
            const activation = MOE_ACTIVATIONS[String(d.activation ?? "gelu").toLowerCase()] ?? "nn.GELU";
            initLines.push(
                `        self.${layerName} = MixtureOfExperts(d_model=${dModel}, d_ff=${dFf}, ` +
                `num_experts=${numExperts}, top_k=${topK}, activation=${activation})`
            );
            definedLayers.add(node.id);
            this.usesMoE = true;
        } else if (!FUNCTIONAL_NODE_TYPES.has(node.type) && node.type !== "constantNode") {
            console.warn(`Node type ${node.type} does not have a manifest entry and is not a recognized functional node.`);
        }
    }

    classDefinition += initLines.join("\n");
    classDefinition += `\n\n`;

    const inputArgs = this.inputNodes.map((node) => this.sanitizeArgName(node.data.name || node.id)).join(", ");
    classDefinition += `    def forward(self, ${inputArgs}):\n`;

    const { lines: forwardLines, nodeOutputs } = this.generateForwardPass(sortedNodes, definedLayers);
    classDefinition += forwardLines.map(line => `        ${line}`).join("\n");

    const outputNode = sortedNodes.find(n => n.type === 'outputNode');
    if (outputNode) {
        const finalEdge = this.edges.find(edge => edge.target === outputNode.id);
        const finalOutputVar = finalEdge ? nodeOutputs.get(finalEdge.source) : undefined;
        if (finalOutputVar) {
            classDefinition += `\n        return ${finalOutputVar}`;
        } else {
            classDefinition += `\n        # No edge connected to output node\n        return None`;
        }
    } else {
        const lastNode = sortedNodes[sortedNodes.length - 1];
        const lastVar = nodeOutputs.get(lastNode.id) ?? this.sanitizeArgName(lastNode.id);
        classDefinition += `\n        return ${lastVar}`;
    }

    // Emit reusable helper modules (e.g. MixtureOfExperts) between the imports
    // and the generated model class, so the generated file is self-contained.
    if (this.usesMoE) {
      code += `\n\n${MOE_HELPER_CLASS}\n`;
    }

    code += classDefinition;

    if (this.inputNodes.length > 0) {
      code += `\n\n\n# Usage example:\n# model = GeneratedModel()\n`;
      const inputVars = [];
      for (const inputNode of this.inputNodes) {
        const varName = this.sanitizeArgName(inputNode.data.name || inputNode.id);
        const { channels, depth, height, width, sequence, length, features } = inputNode.data as any;
        const dims = [channels, depth, height, width, sequence, length, features].filter(
          (d) => typeof d === "number",
        );
        const shape = `(1${dims.map((d) => `, ${d}`).join("")})`;
        code += `# ${varName} = torch.randn${shape}\n`;
        inputVars.push(varName);
      }
      code += `# output = model(${inputVars.join(", ")})\n`;
      code += `# print(f"Output shape: {output.shape}")\n`;
    }
    return code;
  }

  private buildParameterString(node: GraphNode, paramNames: string[]): string {
    const params: string[] = [];
    for (const paramName of paramNames) {
        const value = node.data[paramName];
        if (value !== undefined && value !== null) {
            params.push(`${paramName}=${this.toPythonLiteral(value)}`);
        }
    }
    // Depthwise convolutions require groups=in_channels, which the manifest omits
    if (node.type === "depthwiseconv2dNode" && node.data.in_channels !== undefined) {
        params.push(`groups=${node.data.in_channels}`);
    }
    return params.join(", ");
  }

  private toPythonLiteral(value: any): string {
    if (typeof value === "boolean") {
        return value ? "True" : "False";
    }
    if (Array.isArray(value)) {
        return `(${value.map((v) => this.toPythonLiteral(v)).join(", ")})`;
    }
    if (typeof value === "string") {
        const trimmed = value.trim();
        // Numeric strings (e.g. "3") and comma-separated tuples (e.g. "3, 3")
        // come from free-form UI inputs and must not be quoted.
        if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
            return trimmed;
        }
        if (/^-?\d+(\.\d+)?(\s*,\s*-?\d+(\.\d+)?)+$/.test(trimmed)) {
            return `(${trimmed.split(",").map((s) => s.trim()).join(", ")})`;
        }
        return `"${trimmed}"`;
    }
    return `${value}`;
  }

  private generateForwardPass(
    sortedNodes: GraphNode[],
    definedLayers: Set<string>,
  ): { lines: string[]; nodeOutputs: Map<string, string> } {
    const lines: string[] = [];
    const nodeOutputs = new Map<string, string>();

    for (const inputNode of this.inputNodes) {
      const argName = this.sanitizeArgName(inputNode.data.name || inputNode.id);
      nodeOutputs.set(inputNode.id, argName);
    }

    for (const node of sortedNodes) {
        if (node.type === "inputNode" || node.type === "outputNode") {
            continue;
        }

        const outputVar = this.sanitizeArgName(node.id);

        // Source nodes produce a tensor without any inputs
        if (node.type === "parameterNode") {
            lines.push(`${outputVar} = self.${this.sanitizeArgName(node.id)}`);
            nodeOutputs.set(node.id, outputVar);
            continue;
        }
        if (node.type === "constantNode") {
            const d = node.data as any;
            const dims = [d.channels, d.depth, d.height, d.width, d.sequence, d.length, d.features]
                .filter((v: any) => typeof v === "number");
            lines.push(`${outputVar} = torch.zeros(1${dims.map((v: number) => `, ${v}`).join("")})`);
            nodeOutputs.set(node.id, outputVar);
            continue;
        }

        const inputEdges = this.edges.filter(edge => edge.target === node.id);
        const inputVars = inputEdges
            .map(edge => nodeOutputs.get(edge.source))
            .filter((v): v is string => v !== undefined);

        if (inputVars.length === 0) {
            // Disconnected node: nothing to feed it, skip rather than emit broken code
            continue;
        }

        switch (node.type) {
            case "addNode":
                lines.push(`${outputVar} = ${inputVars.join(" + ")}`);
                break;
            case "multiplyNode":
                lines.push(`${outputVar} = ${inputVars.join(" * ")}`);
                break;
            case "reshapeNode": {
                const shapeLiteral = this.formatTargetShape(node.data.targetShape);
                lines.push(`${outputVar} = ${inputVars[0]}.view(${inputVars[0]}.size(0), *${shapeLiteral})`);
                break;
            }
            case "concatenateNode": {
                const dim = node.data.dim ?? 1;
                lines.push(`${outputVar} = torch.cat([${inputVars.join(", ")}], dim=${dim})`);
                break;
            }
            case "transposeNode": {
                // UI dims are 0-based over non-batch dims; tensor dim 0 is batch
                const dim0 = Number(node.data.dim0 ?? 0) + 1;
                const dim1 = Number(node.data.dim1 ?? 1) + 1;
                lines.push(`${outputVar} = torch.transpose(${inputVars[0]}, ${dim0}, ${dim1})`);
                break;
            }
            case "lstmNode":
            case "gruNode":
            case "rnnNode":
                // Recurrent layers return (output, hidden-state) tuples
                lines.push(`${outputVar}, _ = self.${this.sanitizeArgName(node.id)}(${inputVars[0]})`);
                break;
            case "multiheadattentionNode": {
                // MultiheadAttention takes (query, key, value) and returns (output, weights).
                // With a single input, use self-attention.
                const layerName = this.sanitizeArgName(node.id);
                const [q, k, v] = [
                    inputVars[0],
                    inputVars[1] ?? inputVars[0],
                    inputVars[2] ?? inputVars[1] ?? inputVars[0],
                ];
                lines.push(`${outputVar}, _ = self.${layerName}(${q}, ${k}, ${v})`);
                break;
            }
            default: {
                const layerName = this.sanitizeArgName(node.id);
                if (definedLayers.has(node.id)) {
                    lines.push(`${outputVar} = self.${layerName}(${inputVars.join(', ')})`);
                } else {
                    // Unsupported node type: pass the input through so the code still runs
                    lines.push(`${outputVar} = ${inputVars[0]}  # TODO: '${node.type}' is not supported by the code generator yet`);
                }
                break;
            }
        }
        nodeOutputs.set(node.id, outputVar);
    }
    return { lines, nodeOutputs };
  }

  private formatTargetShape(targetShape: any): string {
    let dims: number[] | null = null;
    if (Array.isArray(targetShape)) {
      dims = targetShape.map(Number).filter((n) => !isNaN(n));
    } else if (typeof targetShape === "string") {
      const cleaned = targetShape.replace(/[[\]()]/g, "").trim();
      if (cleaned) {
        const parsed = cleaned.split(",").map((s) => parseInt(s.trim(), 10));
        if (parsed.every((n) => !isNaN(n))) {
          dims = parsed;
        }
      }
    }
    if (!dims || dims.length === 0) {
      return "(-1,)";
    }
    return `(${dims.join(", ")}${dims.length === 1 ? "," : ""})`;
  }

  private sanitizeArgName(name: string): string {
    return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/\s+/g, "_").toLowerCase();
  }
}
