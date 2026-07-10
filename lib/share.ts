// Shareable-URL encoding for a model graph.
//
// A model graph is a React Flow `{ nodes, edges }` pair. To make a graph
// shareable via a plain link, we serialize it to JSON and pack it into a
// URL-safe base64url string that lives in the URL hash (`#model=...`).
//
// Why the hash and not a query param? The hash is never sent to the server,
// so sharing/restoring stays entirely client-side with no server round-trip.
//
// KNOWN LIMITATION: base64url(JSON) grows with the graph. Very large graphs
// can produce URLs longer than some browsers accept (~2k in older IE/Edge,
// ~8k practical ceiling elsewhere). That is acceptable for now; a future
// version could gzip or fall back to server-side storage for oversized graphs.

export type Graph = { nodes: any[]; edges: any[] }

// --- UTF-8-safe base64url helpers -----------------------------------------
//
// `btoa`/`atob` operate on "binary strings" (one byte per char) and throw on
// characters outside the Latin-1 range. Graphs may contain non-ASCII text
// (unicode node labels, etc.), so we bridge through a UTF-8 byte encoding.
// `TextEncoder`/`TextDecoder` exist in modern browsers and in jsdom/Node, so
// they work in both the app and the vitest test environment.

function bytesToBinaryString(bytes: Uint8Array): string {
  let binary = ""
  // Chunk to avoid "Maximum call stack size exceeded" on large inputs when
  // spreading into String.fromCharCode.
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode.apply(null, chunk as unknown as number[])
  }
  return binary
}

function binaryStringToBytes(binary: string): Uint8Array {
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return bytes
}

function base64Encode(binary: string): string {
  if (typeof btoa === "function") return btoa(binary)
  // Node fallback (should not be needed under jsdom, which provides btoa).
  return Buffer.from(binary, "binary").toString("base64")
}

function base64Decode(b64: string): string {
  if (typeof atob === "function") return atob(b64)
  return Buffer.from(b64, "base64").toString("binary")
}

function toBase64Url(b64: string): string {
  return b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "")
}

function fromBase64Url(b64url: string): string {
  let b64 = b64url.replace(/-/g, "+").replace(/_/g, "/")
  // Restore stripped padding so atob/Buffer can decode.
  const pad = b64.length % 4
  if (pad === 2) b64 += "=="
  else if (pad === 3) b64 += "="
  else if (pad === 1) throw new Error("Invalid base64url length")
  return b64
}

// --- Public API -------------------------------------------------------------

/**
 * Serialize a graph to a URL-safe base64url string (UTF-8 safe).
 */
export function encodeGraph(graph: Graph): string {
  const json = JSON.stringify({ nodes: graph.nodes ?? [], edges: graph.edges ?? [] })
  const bytes = new TextEncoder().encode(json)
  const binary = bytesToBinaryString(bytes)
  return toBase64Url(base64Encode(binary))
}

/**
 * Inverse of `encodeGraph`. Returns null (never throws) on malformed input.
 */
export function decodeGraph(encoded: string): Graph | null {
  if (typeof encoded !== "string" || encoded.length === 0) return null
  try {
    const binary = base64Decode(fromBase64Url(encoded))
    const bytes = binaryStringToBytes(binary)
    const json = new TextDecoder().decode(bytes)
    const parsed = JSON.parse(json)
    if (
      !parsed ||
      typeof parsed !== "object" ||
      !Array.isArray(parsed.nodes) ||
      !Array.isArray(parsed.edges)
    ) {
      return null
    }
    return { nodes: parsed.nodes, edges: parsed.edges }
  } catch {
    return null
  }
}

/**
 * Build a shareable link: `${baseUrl}#model=<encoded>`. The graph rides in the
 * hash so it is never sent to the server.
 */
export function buildShareUrl(baseUrl: string, graph: Graph): string {
  return `${baseUrl}#model=${encodeGraph(graph)}`
}

/**
 * Parse a `#model=...` hash (with or without the leading `#`) and return the
 * decoded graph, or null if the hash is empty, missing `model=`, or malformed.
 */
export function readGraphFromHash(hash: string): Graph | null {
  if (typeof hash !== "string" || hash.length === 0) return null
  const raw = hash.startsWith("#") ? hash.slice(1) : hash
  if (raw.length === 0) return null
  // The hash may carry other params (`a=b&model=...`); scan each segment.
  for (const part of raw.split("&")) {
    if (part.startsWith("model=")) {
      return decodeGraph(part.slice("model=".length))
    }
  }
  return null
}
