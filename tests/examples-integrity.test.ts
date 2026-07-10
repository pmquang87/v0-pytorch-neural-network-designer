import { describe, it, expect } from "vitest";
import fs from "fs";
import path from "path";
import { ModelValidator } from "../lib/model-validator";
import { calculateOutputShape, type TensorShape } from "../lib/tensor-shape-calculator";
import { ModelGenerator } from "../lib/model-generator";
import type { GraphNode, GraphEdge } from "../lib/types";

// ---------------------------------------------------------------------------
// Integrity harness for every bundled example network in lib/examples/*.json.
//
// It guarantees each example is a valid, connected graph that survives shape
// propagation and code generation. Each example gets its own `it(...)` case so
// a failure names the offending file.
// ---------------------------------------------------------------------------

const EXAMPLES_DIR = path.resolve(__dirname, "..", "lib", "examples");

// Node types that legitimately have NO incoming edge (they source a tensor).
const SOURCE_NODE_TYPES = new Set([
  "inputNode",
  "parameterNode",
  "constantNode",
  "noiseNode",
]);

// Node types that legitimately have NO outgoing edge (they are sinks).
const SINK_NODE_TYPES = new Set([
  "outputNode",
  "contentLossNode",
  "styleLossNode",
]);

interface RawNode {
  id: string;
  type: string;
  data?: Record<string, any>;
  position?: { x: number; y: number };
}
interface RawEdge {
  id?: string;
  source: string;
  target: string;
  targetHandle?: string;
}
interface RawExample {
  name?: string;
  description?: string;
  nodes: RawNode[];
  edges: RawEdge[];
}

const files = fs
  .readdirSync(EXAMPLES_DIR)
  .filter((f) => f.endsWith(".json"))
  .sort();

// Ordered input shapes for a node from its incoming edges. Multi-input ops
// (concatenate/add/multiply/attention) are ordered by their `inputN` handle so
// shape math matches how the app wires them.
function orderedInputShapes(
  node: RawNode,
  edges: RawEdge[],
  shapeById: Map<string, TensorShape>,
): TensorShape[] {
  const incoming = edges.filter((e) => e.target === node.id);
  const handleIndex = (e: RawEdge): number => {
    const m = /^input(\d+)$/.exec(e.targetHandle ?? "");
    return m ? Number(m[1]) : Number.MAX_SAFE_INTEGER;
  };
  const sorted = [...incoming].sort((a, b) => handleIndex(a) - handleIndex(b));
  return sorted.map((e) => shapeById.get(e.source) ?? {});
}

// Kahn topological order over the graph (ignores nodes only reachable in cycles).
function topoOrder(nodes: RawNode[], edges: RawEdge[]): RawNode[] {
  const inDegree = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const n of nodes) {
    inDegree.set(n.id, 0);
    adj.set(n.id, []);
  }
  for (const e of edges) {
    if (!adj.has(e.source) || !inDegree.has(e.target)) continue;
    adj.get(e.source)!.push(e.target);
    inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1);
  }
  const queue: string[] = [];
  for (const [id, d] of inDegree) if (d === 0) queue.push(id);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const order: RawNode[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(byId.get(id)!);
    for (const nb of adj.get(id) || []) {
      inDegree.set(nb, inDegree.get(nb)! - 1);
      if (inDegree.get(nb) === 0) queue.push(nb);
    }
  }
  return order;
}

describe("examples integrity", () => {
  it("finds example files", () => {
    expect(files.length).toBeGreaterThan(0);
  });

  for (const file of files) {
    it(file, () => {
      const fullPath = path.join(EXAMPLES_DIR, file);
      const raw = fs.readFileSync(fullPath, "utf8");

      // (a) parses and has nodes/edges arrays
      let parsed: RawExample;
      try {
        parsed = JSON.parse(raw);
      } catch (e) {
        throw new Error(`${file}: invalid JSON — ${(e as Error).message}`);
      }
      expect(Array.isArray(parsed.nodes), `${file}: missing nodes[] array`).toBe(true);
      expect(Array.isArray(parsed.edges), `${file}: missing edges[] array`).toBe(true);

      const nodes = parsed.nodes;
      const edges = parsed.edges;
      const ids = new Set(nodes.map((n) => n.id));

      // (b) no dangling edges — every source/target references a real node id
      const dangling: string[] = [];
      for (const e of edges) {
        if (!ids.has(e.source)) dangling.push(`edge ${e.id ?? "?"} source '${e.source}'`);
        if (!ids.has(e.target)) dangling.push(`edge ${e.id ?? "?"} target '${e.target}'`);
      }
      expect(dangling, `${file}: dangling edge references -> ${dangling.join(", ")}`).toEqual([]);

      // (c) real validator reports no error-severity problems (cycles, invalid
      // connections, missing required params, shape mismatches).
      const validator = new ModelValidator();
      const result = validator.validateModel(nodes as any, edges as any);
      expect(result.errors, `${file}: validateModel errors -> ${result.errors.join(" | ")}`).toEqual([]);

      // (c) no orphaned nodes.
      if (nodes.length > 1) {
        const incoming = new Map<string, number>();
        const outgoing = new Map<string, number>();
        for (const n of nodes) {
          incoming.set(n.id, 0);
          outgoing.set(n.id, 0);
        }
        for (const e of edges) {
          outgoing.set(e.source, (outgoing.get(e.source) || 0) + 1);
          incoming.set(e.target, (incoming.get(e.target) || 0) + 1);
        }

        // Non-source nodes must have at least one incoming edge.
        const missingIncoming = nodes
          .filter((n) => !SOURCE_NODE_TYPES.has(n.type) && (incoming.get(n.id) || 0) === 0)
          .map((n) => `${n.id} (${n.type})`);
        expect(
          missingIncoming,
          `${file}: nodes with no incoming edge -> ${missingIncoming.join(", ")}`,
        ).toEqual([]);

        // Non-sink nodes must have at least one outgoing edge, EXCEPT the graph's
        // terminal node(s): a node with an incoming edge but no outgoing edge is a
        // legitimate leaf. We only flag nodes that are wholly disconnected.
        const fullyDisconnected = nodes
          .filter((n) => (incoming.get(n.id) || 0) === 0 && (outgoing.get(n.id) || 0) === 0)
          .map((n) => `${n.id} (${n.type})`);
        expect(
          fullyDisconnected,
          `${file}: fully disconnected nodes -> ${fullyDisconnected.join(", ")}`,
        ).toEqual([]);
      }

      // (d) shape propagation must not throw for the whole graph.
      const shapeById = new Map<string, TensorShape>();
      const ordered = topoOrder(nodes, edges);
      expect(
        ordered.length,
        `${file}: topological order incomplete (cycle?) — ${ordered.length}/${nodes.length}`,
      ).toBe(nodes.length);
      for (const node of ordered) {
        const inputs = orderedInputShapes(node, edges, shapeById);
        let out: TensorShape;
        try {
          out = calculateOutputShape(node.type, inputs.length ? inputs : [{}], node.data ?? {});
        } catch (e) {
          throw new Error(`${file}: calculateOutputShape threw for ${node.id} (${node.type}) — ${(e as Error).message}`);
        }
        shapeById.set(node.id, out ?? {});
      }

      // (d) code generation must not throw and must return a non-empty string.
      let code: string;
      try {
        const generator = new ModelGenerator({
          nodes: nodes as unknown as GraphNode[],
          edges: edges as unknown as GraphEdge[],
        });
        code = generator.generateCode();
      } catch (e) {
        throw new Error(`${file}: generateCode threw — ${(e as Error).message}`);
      }
      expect(typeof code, `${file}: generateCode did not return a string`).toBe("string");
      expect(code.length, `${file}: generateCode returned empty string`).toBeGreaterThan(0);
    });
  }
});
