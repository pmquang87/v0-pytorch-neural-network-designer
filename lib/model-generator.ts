
import { type GraphIR, type GraphNode, type GraphEdge, PYTORCH_LAYER_MANIFEST } from "./types";

export class ModelGenerator {
  private readonly nodes: GraphNode[];
  private readonly edges: GraphEdge[];
  private readonly inputNodes: GraphNode[] = [];

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
    for (const node of layerNodes) {
        const manifest = PYTORCH_LAYER_MANIFEST[node.type as keyof typeof PYTORCH_LAYER_MANIFEST];
        const layerName = this.sanitizeArgName(node.id);

        if (manifest && manifest.className) {
            const params = this.buildParameterString(node, manifest.params);
            initLines.push(`        self.${layerName} = ${manifest.className}(${params})`);
        } else if (!["reshapeNode", "addNode", "multiplyNode", "concatenateNode"].includes(node.type)) {
            console.warn(`Node type ${node.type} does not have a manifest entry and is not a recognized functional node.`);
        }
    }

    classDefinition += initLines.join("\n");
    classDefinition += `\n\n`;

    const inputArgs = this.inputNodes.map((node) => this.sanitizeArgName(node.data.name || node.id)).join(", ");
    classDefinition += `    def forward(self, ${inputArgs}):\n`;

    const forwardLinesResult = this.generateForwardPass(sortedNodes);
    classDefinition += forwardLinesResult.map(line => `        ${line}`).join("\n");

    const outputNode = sortedNodes.find(n => n.type === 'outputNode');
    if (outputNode) {
        const finalEdge = this.edges.find(edge => edge.target === outputNode.id);
        if (finalEdge) {
            const finalOutputVar = this.sanitizeArgName(finalEdge.source);
            classDefinition += `\n        return ${finalOutputVar}`;
        } else {
            classDefinition += `\n        # No edge connected to output node\n        return None`;
        }
    } else {
        const lastNode = sortedNodes[sortedNodes.length - 1];
        classDefinition += `\n        return ${this.sanitizeArgName(lastNode.id)}`;
    }

    code += classDefinition;

    if (this.inputNodes.length > 0) {
      code += `\n\n\n# Usage example:\n# model = GeneratedModel()\n`;
      const inputVars = [];
      for (const inputNode of this.inputNodes) {
        const varName = this.sanitizeArgName(inputNode.data.name || inputNode.id);
        const { channels, height, width, features } = inputNode.data;
        let shape;
        if (height !== undefined && width !== undefined) {
          shape = `(1, ${channels}, ${height}, ${width})`;
        } else if (features !== undefined) {
          shape = `(1, ${features})`;
        } else {
          shape = `(1, ${channels})`;
        }
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
            if (Array.isArray(value)) {
                params.push(`${paramName}=${JSON.stringify(value)}`);
            } else if (typeof value === "string") {
                params.push(`${paramName}="${value}"`);
            } else {
                params.push(`${paramName}=${value}`);
            }
        }
    }
    return params.join(", ");
  }

  private generateForwardPass(sortedNodes: GraphNode[]): string[] {
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
        const inputEdges = this.edges.filter(edge => edge.target === node.id);
        const inputVars = inputEdges.map(edge => nodeOutputs.get(edge.source)!);

        switch (node.type) {
            case "addNode":
                lines.push(`${outputVar} = ${inputVars.join(" + ")}`);
                break;
            case "multiplyNode":
                lines.push(`${outputVar} = ${inputVars.join(" * ")}`);
                break;
            case "reshapeNode":
                const shape = node.data.targetShape;
                lines.push(`${outputVar} = ${inputVars[0]}.view(${inputVars[0]}.size(0), *${JSON.stringify(shape)})`);
                break;
            case "concatenateNode":
                const dim = node.data.dim || 1;
                lines.push(`${outputVar} = torch.cat([${inputVars.join(", ")}], dim=${dim})`);
                break;
            default:
                const layerName = this.sanitizeArgName(node.id);
                lines.push(`${outputVar} = self.${layerName}(${inputVars.join(', ')})`);
                break;
        }
        nodeOutputs.set(node.id, outputVar);
    }
    return lines;
  }

  private sanitizeArgName(name: string): string {
    return name.replace(/[^a-zA-Z0-9_]/g, "_").replace(/\s+/g, "_").toLowerCase();
  }
}
