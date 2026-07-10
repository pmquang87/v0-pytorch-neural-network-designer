import { describe, it, expect } from "vitest"
import {
  encodeGraph,
  decodeGraph,
  buildShareUrl,
  readGraphFromHash,
  type Graph,
} from "../lib/share"

const sampleGraph: Graph = {
  nodes: [
    {
      id: "input-1",
      type: "inputNode",
      position: { x: 100, y: 200 },
      data: { channels: 1, height: 28, width: 28 },
    },
    {
      id: "linear-1",
      type: "linearNode",
      position: { x: 500, y: 200 },
      data: { in_features: 784, out_features: 256, name: "fc1" },
    },
  ],
  edges: [{ id: "e1", source: "input-1", target: "linear-1" }],
}

describe("encodeGraph / decodeGraph", () => {
  it("round-trips a graph exactly", () => {
    const encoded = encodeGraph(sampleGraph)
    expect(typeof encoded).toBe("string")
    const decoded = decodeGraph(encoded)
    expect(decoded).toEqual(sampleGraph)
  })

  it("produces a URL-safe base64url string (no +, /, or =)", () => {
    // Build a graph large enough to force base64 padding/special chars.
    const big: Graph = {
      nodes: Array.from({ length: 50 }, (_, i) => ({
        id: `n${i}`,
        type: "linearNode",
        data: { in_features: i, out_features: i + 1, name: `layer ${i} >>> /\\?` },
      })),
      edges: [],
    }
    const encoded = encodeGraph(big)
    expect(encoded).not.toMatch(/[+/=]/)
    expect(decodeGraph(encoded)).toEqual(big)
  })

  it("round-trips a graph with unicode in a node label", () => {
    const unicodeGraph: Graph = {
      nodes: [
        {
          id: "u1",
          type: "linearNode",
          data: { name: "层 café 😀 Ω → ✓ Ελληνικά", note: "日本語テスト" },
        },
      ],
      edges: [],
    }
    const encoded = encodeGraph(unicodeGraph)
    const decoded = decodeGraph(encoded)
    expect(decoded).toEqual(unicodeGraph)
    expect(decoded?.nodes[0].data.name).toBe("层 café 😀 Ω → ✓ Ελληνικά")
  })

  it("round-trips an empty graph", () => {
    const empty: Graph = { nodes: [], edges: [] }
    const decoded = decodeGraph(encodeGraph(empty))
    expect(decoded).toEqual(empty)
  })

  it("returns null for empty string", () => {
    expect(decodeGraph("")).toBeNull()
  })

  it("returns null for malformed input instead of throwing", () => {
    expect(decodeGraph("!!!not base64!!!")).toBeNull()
    expect(decodeGraph("####")).toBeNull()
    // Valid base64url but not JSON.
    expect(decodeGraph(encodeGraph({ nodes: [], edges: [] }) + "garbage")).toBeNull()
  })

  it("returns null when decoded JSON is not a graph shape", () => {
    // Encode a plain array (valid JSON, wrong shape).
    const notAGraph = encodeGraph({ nodes: undefined as any, edges: undefined as any })
    // encodeGraph defaults undefined to [], so this actually is valid; test a
    // hand-rolled encoding of a non-graph object instead.
    const encodedNumber = (() => {
      const json = JSON.stringify(42)
      const bytes = new TextEncoder().encode(json)
      let binary = ""
      bytes.forEach((b) => (binary += String.fromCharCode(b)))
      return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "")
    })()
    expect(decodeGraph(encodedNumber)).toBeNull()
    expect(decodeGraph(notAGraph)).toEqual({ nodes: [], edges: [] })
  })
})

describe("buildShareUrl / readGraphFromHash", () => {
  it("builds a #model= url and reads it back", () => {
    const url = buildShareUrl("https://example.com/designer", sampleGraph)
    expect(url.startsWith("https://example.com/designer#model=")).toBe(true)
    const hash = url.slice(url.indexOf("#"))
    expect(readGraphFromHash(hash)).toEqual(sampleGraph)
  })

  it("accepts a hash with or without the leading #", () => {
    const encoded = encodeGraph(sampleGraph)
    expect(readGraphFromHash(`#model=${encoded}`)).toEqual(sampleGraph)
    expect(readGraphFromHash(`model=${encoded}`)).toEqual(sampleGraph)
  })

  it("finds model= among other hash params", () => {
    const encoded = encodeGraph(sampleGraph)
    expect(readGraphFromHash(`#tab=code&model=${encoded}&x=1`)).toEqual(sampleGraph)
  })

  it("returns null for empty hash", () => {
    expect(readGraphFromHash("")).toBeNull()
    expect(readGraphFromHash("#")).toBeNull()
  })

  it("returns null for a hash without model=", () => {
    expect(readGraphFromHash("#tab=code&foo=bar")).toBeNull()
  })

  it("returns null when model payload is malformed", () => {
    expect(readGraphFromHash("#model=!!!bad!!!")).toBeNull()
  })
})
