// PyTorch source -> complete computation graph.
//
// Unlike a naive importer that only reads `__init__` and wires the declared
// layers into a straight line, this parser reads the model's `forward()` method
// and reconstructs the *actual* data-flow graph: residual/skip connections,
// concatenations, element-wise merges, functional activations, tensor reshapes,
// reused layers, and (recursively) user-defined submodules. The result is a set
// of React-Flow-compatible nodes and edges that faithfully represent the model.

import type { GraphNode, GraphEdge } from "./types"

export interface ParsedNode {
  id: string
  type: string
  data: Record<string, any>
  position: { x: number; y: number }
}

export interface ParsedEdge {
  id: string
  source: string
  target: string
  type: string
  targetHandle?: string
}

export interface ParseResult {
  nodes: ParsedNode[]
  edges: ParsedEdge[]
  errors: string[]
  warnings: string[]
  unsupportedModules: string[]
}

// ---------------------------------------------------------------------------
// nn.Module class name -> canvas node type
// ---------------------------------------------------------------------------
export const NN_NODE_TYPE_MAP: Record<string, string> = {
  Linear: "linearNode",
  LazyLinear: "linearNode",
  Bilinear: "linearNode",
  Conv1d: "conv1dNode",
  Conv2d: "conv2dNode",
  Conv3d: "conv3dNode",
  ConvTranspose1d: "convtranspose1dNode",
  ConvTranspose2d: "convtranspose2dNode",
  ConvTranspose3d: "convtranspose3dNode",
  BatchNorm1d: "batchnorm1dNode",
  BatchNorm2d: "batchnorm2dNode",
  BatchNorm3d: "batchnorm3dNode",
  LayerNorm: "layernormNode",
  RMSNorm: "rmsnormNode",
  GroupNorm: "groupnormNode",
  InstanceNorm1d: "instancenorm1dNode",
  InstanceNorm2d: "instancenorm2dNode",
  InstanceNorm3d: "instancenorm3dNode",
  ReLU: "reluNode",
  ReLU6: "reluNode",
  LeakyReLU: "leakyreluNode",
  GELU: "geluNode",
  SiLU: "siluNode",
  Mish: "mishNode",
  Hardswish: "hardswishNode",
  Hardsigmoid: "hardsigmoidNode",
  Tanh: "tanhNode",
  Sigmoid: "sigmoidNode",
  Softmax: "softmaxNode",
  LogSoftmax: "softmaxNode",
  MaxPool1d: "maxpool1dNode",
  MaxPool2d: "maxpool2dNode",
  MaxPool3d: "maxpool3dNode",
  AvgPool2d: "avgpool2dNode",
  AdaptiveAvgPool2d: "adaptiveavgpool2dNode",
  AdaptiveMaxPool1d: "adaptivemaxpool1dNode",
  FractionalMaxPool2d: "fractionalmaxpool2dNode",
  LPPool2d: "lppool2dNode",
  Dropout: "dropoutNode",
  Dropout2d: "dropoutNode",
  Dropout3d: "dropoutNode",
  Flatten: "flattenNode",
  LSTM: "lstmNode",
  GRU: "gruNode",
  RNN: "rnnNode",
  MultiheadAttention: "multiheadattentionNode",
  TransformerEncoderLayer: "transformerencoderlayerNode",
  TransformerDecoderLayer: "transformerdecoderlayerNode",
  Upsample: "upsampleNode",
}

// Node types that accept more than one distinct tensor input and therefore
// need ordered `input1`, `input2`, ... target handles.
const MULTI_INPUT_NODE_TYPES = new Set(["addNode", "multiplyNode", "concatenateNode"])

// Recurrent / attention modules return a tuple (output, state); the first
// element carries the tensor we follow through the graph.
const TUPLE_OUTPUT_NODE_TYPES = new Set(["lstmNode", "gruNode", "rnnNode", "multiheadattentionNode"])

// Functional ops (F.* / torch.* / tensor methods) -> canvas node type.
const FUNCTIONAL_NODE_TYPE_MAP: Record<string, string> = {
  relu: "reluNode",
  relu6: "reluNode",
  leaky_relu: "leakyreluNode",
  gelu: "geluNode",
  silu: "siluNode",
  mish: "mishNode",
  hardswish: "hardswishNode",
  hardsigmoid: "hardsigmoidNode",
  tanh: "tanhNode",
  sigmoid: "sigmoidNode",
  softmax: "softmaxNode",
  log_softmax: "softmaxNode",
  max_pool2d: "maxpool2dNode",
  avg_pool2d: "avgpool2dNode",
  adaptive_avg_pool2d: "adaptiveavgpool2dNode",
  dropout: "dropoutNode",
  flatten: "flattenNode",
}

// ---------------------------------------------------------------------------
// Expression AST + Pratt parser for the subset of Python we care about.
// ---------------------------------------------------------------------------
type Expr =
  | { kind: "name"; id: string }
  | { kind: "num"; value: number }
  | { kind: "str"; value: string }
  | { kind: "const"; value: string } // True/False/None/other bare words
  | { kind: "attr"; obj: Expr; attr: string }
  | { kind: "index"; obj: Expr; index: Expr | null }
  | { kind: "call"; func: Expr; args: Expr[]; kwargs: Record<string, Expr> }
  | { kind: "binop"; op: string; left: Expr; right: Expr }
  | { kind: "unary"; op: string; operand: Expr }
  | { kind: "list"; elts: Expr[] }
  | { kind: "tuple"; elts: Expr[] }

interface Token {
  type: "name" | "num" | "str" | "op"
  value: string
}

function tokenizeExpr(src: string): Token[] {
  const tokens: Token[] = []
  let i = 0
  const n = src.length
  while (i < n) {
    const c = src[i]
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      i++
      continue
    }
    // strings
    if (c === '"' || c === "'") {
      const quote = c
      let j = i + 1
      let val = ""
      while (j < n && src[j] !== quote) {
        if (src[j] === "\\" && j + 1 < n) {
          val += src[j + 1]
          j += 2
        } else {
          val += src[j]
          j++
        }
      }
      tokens.push({ type: "str", value: val })
      i = j + 1
      continue
    }
    // numbers (int, float, scientific)
    if (/[0-9]/.test(c) || (c === "." && /[0-9]/.test(src[i + 1] || ""))) {
      let j = i
      while (j < n && /[0-9.eE+\-_]/.test(src[j])) {
        // stop a trailing sign that isn't part of an exponent
        if ((src[j] === "+" || src[j] === "-") && !/[eE]/.test(src[j - 1] || "")) break
        j++
      }
      tokens.push({ type: "num", value: src.slice(i, j) })
      i = j
      continue
    }
    // identifiers / keywords
    if (/[A-Za-z_]/.test(c)) {
      let j = i
      while (j < n && /[A-Za-z0-9_]/.test(src[j])) j++
      tokens.push({ type: "name", value: src.slice(i, j) })
      i = j
      continue
    }
    // two-char operators
    const two = src.slice(i, i + 2)
    if (["**", "//", "==", "!=", ">=", "<="].includes(two)) {
      tokens.push({ type: "op", value: two })
      i += 2
      continue
    }
    tokens.push({ type: "op", value: c })
    i++
  }
  return tokens
}

// Recursive-descent parser over the token stream.
class ExprParser {
  private toks: Token[]
  private pos = 0
  constructor(toks: Token[]) {
    this.toks = toks
  }
  private peek(): Token | undefined {
    return this.toks[this.pos]
  }
  private next(): Token | undefined {
    return this.toks[this.pos++]
  }
  private expect(value: string) {
    const t = this.next()
    if (!t || t.value !== value) {
      throw new Error(`Expected '${value}' but got '${t?.value ?? "<eof>"}'`)
    }
  }

  parse(): Expr {
    const e = this.parseExpr(0)
    return e
  }

  // Pratt-style binary parsing.
  private parseExpr(minBp: number): Expr {
    let left = this.parseUnary()
    while (true) {
      const t = this.peek()
      if (!t || t.type !== "op") break
      const bp = BINDING_POWER[t.value]
      if (bp === undefined || bp < minBp) break
      this.next()
      const right = this.parseExpr(bp + 1)
      left = { kind: "binop", op: t.value, left, right }
    }
    return left
  }

  private parseUnary(): Expr {
    const t = this.peek()
    if (t && t.type === "op" && (t.value === "-" || t.value === "+" || t.value === "~")) {
      this.next()
      return { kind: "unary", op: t.value, operand: this.parseUnary() }
    }
    if (t && t.type === "name" && t.value === "not") {
      this.next()
      return { kind: "unary", op: "not", operand: this.parseUnary() }
    }
    return this.parsePostfix()
  }

  private parsePostfix(): Expr {
    let e = this.parseAtom()
    while (true) {
      const t = this.peek()
      if (!t) break
      if (t.value === ".") {
        this.next()
        const name = this.next()
        if (!name || name.type !== "name") throw new Error("Expected attribute name after '.'")
        e = { kind: "attr", obj: e, attr: name.value }
      } else if (t.value === "(") {
        e = this.parseCall(e)
      } else if (t.value === "[") {
        this.next()
        let index: Expr | null = null
        if (this.peek()?.value !== "]") {
          // Only capture a simple index; slices collapse to null.
          try {
            index = this.parseExpr(0)
          } catch {
            index = null
          }
          // skip anything up to the matching ]
          this.skipUntil("]")
          this.pos-- // step back so expect consumes ']'
        }
        this.expect("]")
        e = { kind: "index", obj: e, index }
      } else {
        break
      }
    }
    return e
  }

  private skipUntil(closer: string) {
    let depth = 0
    while (this.pos < this.toks.length) {
      const v = this.toks[this.pos].value
      if (v === "(" || v === "[" || v === "{") depth++
      else if (v === ")" || v === "]" || v === "}") {
        if (depth === 0 && v === closer) return
        depth--
      }
      this.pos++
    }
  }

  private parseCall(func: Expr): Expr {
    this.expect("(")
    const args: Expr[] = []
    const kwargs: Record<string, Expr> = {}
    while (this.peek() && this.peek()!.value !== ")") {
      // keyword argument?  name '=' expr
      const t = this.peek()!
      const after = this.toks[this.pos + 1]
      if (t.type === "name" && after && after.value === "=") {
        this.next() // name
        this.next() // '='
        kwargs[t.value] = this.parseExpr(0)
      } else {
        args.push(this.parseExpr(0))
      }
      if (this.peek()?.value === ",") this.next()
      else break
    }
    this.expect(")")
    return { kind: "call", func, args, kwargs }
  }

  private parseAtom(): Expr {
    const t = this.next()
    if (!t) throw new Error("Unexpected end of expression")
    if (t.type === "num") {
      const val = Number(t.value.replace(/_/g, ""))
      return { kind: "num", value: isNaN(val) ? 0 : val }
    }
    if (t.type === "str") return { kind: "str", value: t.value }
    if (t.type === "name") {
      if (t.value === "True" || t.value === "False" || t.value === "None") {
        return { kind: "const", value: t.value }
      }
      return { kind: "name", id: t.value }
    }
    if (t.value === "(") {
      // grouping or tuple
      const elts: Expr[] = []
      let isTuple = false
      if (this.peek()?.value !== ")") {
        elts.push(this.parseExpr(0))
        while (this.peek()?.value === ",") {
          isTuple = true
          this.next()
          if (this.peek()?.value === ")") break
          elts.push(this.parseExpr(0))
        }
      }
      this.expect(")")
      return isTuple ? { kind: "tuple", elts } : elts[0] ?? { kind: "tuple", elts: [] }
    }
    if (t.value === "[") {
      const elts: Expr[] = []
      while (this.peek() && this.peek()!.value !== "]") {
        elts.push(this.parseExpr(0))
        if (this.peek()?.value === ",") this.next()
        else break
      }
      this.expect("]")
      return { kind: "list", elts }
    }
    if (t.value === "{") {
      // dict / set — not needed for graph building; skip its contents.
      this.pos--
      this.skipUntil("}")
      this.expect("}")
      return { kind: "const", value: "None" }
    }
    // Unknown token; treat as bare constant so parsing keeps moving.
    return { kind: "const", value: t.value }
  }
}

const BINDING_POWER: Record<string, number> = {
  "+": 10,
  "-": 10,
  "*": 20,
  "/": 20,
  "//": 20,
  "%": 20,
  "@": 20,
  "**": 30,
}

function parseExpression(src: string): Expr | null {
  try {
    const toks = tokenizeExpr(src)
    if (toks.length === 0) return null
    return new ExprParser(toks).parse()
  } catch {
    return null
  }
}

// ---------------------------------------------------------------------------
// Parameter value coercion (Python literal -> JS value).
// ---------------------------------------------------------------------------
function exprToValue(e: Expr | undefined): any {
  if (!e) return undefined
  switch (e.kind) {
    case "num":
      return e.value
    case "str":
      return e.value
    case "const":
      if (e.value === "True") return true
      if (e.value === "False") return false
      if (e.value === "None") return null
      return e.value
    case "unary":
      if (e.op === "-") {
        const v = exprToValue(e.operand)
        return typeof v === "number" ? -v : v
      }
      return exprToValue(e.operand)
    case "tuple":
    case "list": {
      const vals = e.elts.map(exprToValue)
      return vals.length === 1 ? vals[0] : vals
    }
    default:
      return undefined
  }
}

// ---------------------------------------------------------------------------
// __init__ module descriptors.
// ---------------------------------------------------------------------------
type ModuleDescriptor =
  | { kind: "layer"; layerType: string; args: Expr[]; kwargs: Record<string, Expr> }
  | { kind: "sequential"; items: ModuleDescriptor[] }
  | { kind: "modulelist"; items: ModuleDescriptor[] }
  | { kind: "custom"; className: string; args: Expr[]; kwargs: Record<string, Expr> }
  | { kind: "unknown"; label: string }

interface ParsedClass {
  name: string
  modules: Map<string, ModuleDescriptor>
  forwardArgs: string[]
  forwardBody: string
}

// ---------------------------------------------------------------------------
// Source preprocessing helpers.
// ---------------------------------------------------------------------------
function stripComments(code: string): string {
  const out: string[] = []
  for (const rawLine of code.split("\n")) {
    let inStr = false
    let quote = ""
    let result = ""
    for (let i = 0; i < rawLine.length; i++) {
      const ch = rawLine[i]
      if (inStr) {
        result += ch
        if (ch === quote && rawLine[i - 1] !== "\\") inStr = false
      } else if (ch === '"' || ch === "'") {
        inStr = true
        quote = ch
        result += ch
      } else if (ch === "#") {
        break
      } else {
        result += ch
      }
    }
    out.push(result)
  }
  return out.join("\n")
}

// Extract the body of a `def name(...)` block using indentation.
function extractBlock(source: string, header: RegExp): { args: string; body: string } | null {
  const lines = source.split("\n")
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(header)
    if (!m) continue
    // A def header can span multiple lines until the ':' at bracket depth 0.
    let headerText = lines[i]
    let j = i
    while (!/:\s*$/.test(stripTrailing(headerText)) && j + 1 < lines.length) {
      j++
      headerText += "\n" + lines[j]
    }
    const argsMatch = headerText.match(/\(([\s\S]*)\)\s*:/)
    const args = argsMatch ? argsMatch[1] : ""
    const defIndent = lines[i].match(/^(\s*)/)![1].length
    const bodyLines: string[] = []
    for (let k = j + 1; k < lines.length; k++) {
      const line = lines[k]
      if (line.trim() === "") {
        bodyLines.push(line)
        continue
      }
      const indent = line.match(/^(\s*)/)![1].length
      if (indent <= defIndent) break
      bodyLines.push(line)
    }
    return { args, body: bodyLines.join("\n") }
  }
  return null
}

function stripTrailing(s: string): string {
  return s.replace(/\s+$/, "")
}

// Split a class body into logical statements, joining lines that are inside
// open brackets or that use an explicit backslash continuation.
function splitStatements(body: string): string[] {
  const statements: string[] = []
  let current = ""
  let depth = 0
  let inStr = false
  let quote = ""
  const lines = body.split("\n")
  for (let li = 0; li < lines.length; li++) {
    let line = lines[li]
    for (let i = 0; i < line.length; i++) {
      const ch = line[i]
      if (inStr) {
        if (ch === quote && line[i - 1] !== "\\") inStr = false
      } else if (ch === '"' || ch === "'") {
        inStr = true
        quote = ch
      } else if (ch === "(" || ch === "[" || ch === "{") depth++
      else if (ch === ")" || ch === "]" || ch === "}") depth = Math.max(0, depth - 1)
    }
    const backslash = line.endsWith("\\")
    current += (current ? "\n" : "") + (backslash ? line.slice(0, -1) : line)
    if (depth === 0 && !backslash && !inStr) {
      if (current.trim()) statements.push(current.trim())
      current = ""
    }
  }
  if (current.trim()) statements.push(current.trim())
  // Also split simple `a; b` on the same line.
  const final: string[] = []
  for (const s of statements) {
    if (s.includes(";") && !s.includes('"') && !s.includes("'")) {
      for (const part of s.split(";")) if (part.trim()) final.push(part.trim())
    } else {
      final.push(s)
    }
  }
  return final
}

// ---------------------------------------------------------------------------
// Class extraction.
// ---------------------------------------------------------------------------
function extractClasses(code: string): Map<string, ParsedClass> {
  const classes = new Map<string, ParsedClass>()
  const lines = code.split("\n")
  const classHeader = /^(\s*)class\s+(\w+)\s*(\(([^)]*)\))?\s*:/
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(classHeader)
    if (!m) continue
    const indent = m[1].length
    const className = m[2]
    const bases = m[4] || ""
    // Gather the class body by indentation.
    const bodyLines: string[] = []
    for (let k = i + 1; k < lines.length; k++) {
      const line = lines[k]
      if (line.trim() === "") {
        bodyLines.push(line)
        continue
      }
      const li = line.match(/^(\s*)/)![1].length
      if (li <= indent) break
      bodyLines.push(line)
    }
    const body = bodyLines.join("\n")
    const isModule = /Module|Block|Net|Layer|Model/.test(bases) || /nn\.Module/.test(bases) || bases.trim() !== ""
    const initBlock = extractBlock(body, /^\s*def\s+__init__\s*\(/)
    const forwardBlock = extractBlock(body, /^\s*def\s+forward\s*\(/)
    const modules = initBlock ? parseInitModules(initBlock.body) : new Map<string, ModuleDescriptor>()
    let forwardArgs: string[] = []
    if (forwardBlock) {
      forwardArgs = forwardBlock.args
        .split(",")
        .map((a) => a.trim().split(/[:=]/)[0].trim())
        .filter((a) => a && a !== "self")
    }
    classes.set(className, {
      name: className,
      modules,
      forwardArgs,
      forwardBody: forwardBlock ? forwardBlock.body : "",
      // @ts-expect-error carry a hint used only for main-class selection
      _isModule: isModule,
    })
  }
  return classes
}

// Parse `self.x = nn.Xxx(...)` / Sequential / ModuleList / custom submodules.
function parseInitModules(initBody: string): Map<string, ModuleDescriptor> {
  const modules = new Map<string, ModuleDescriptor>()
  for (const stmt of splitStatements(initBody)) {
    const m = stmt.match(/^self\.(\w+)\s*=\s*([\s\S]+)$/)
    if (!m) continue
    const name = m[1]
    const rhs = m[2].trim()
    const desc = describeModuleExpr(rhs)
    if (desc) modules.set(name, desc)
  }
  return modules
}

function describeModuleExpr(rhs: string): ModuleDescriptor | null {
  const expr = parseExpression(rhs)
  if (!expr) return null
  return describeModuleFromExpr(expr)
}

function describeModuleFromExpr(expr: Expr): ModuleDescriptor | null {
  if (expr.kind !== "call") return null
  const callee = calleeName(expr.func)
  if (!callee) return null

  // nn.Sequential(a, b, ...) or nn.Sequential([a, b, ...])
  if (callee === "Sequential" || callee.endsWith(".Sequential")) {
    const items: ModuleDescriptor[] = []
    const elements = expr.args.length === 1 && (expr.args[0].kind === "list" || expr.args[0].kind === "tuple")
      ? (expr.args[0] as any).elts
      : expr.args
    for (const el of elements) {
      const d = describeModuleFromExpr(el)
      if (d) items.push(d)
    }
    return { kind: "sequential", items }
  }

  // nn.ModuleList([...])
  if (callee === "ModuleList" || callee.endsWith(".ModuleList")) {
    const items: ModuleDescriptor[] = []
    const listArg = expr.args[0]
    if (listArg && (listArg.kind === "list" || listArg.kind === "tuple")) {
      for (const el of listArg.elts) {
        const d = describeModuleFromExpr(el)
        if (d) items.push(d)
      }
    }
    return { kind: "modulelist", items }
  }

  // nn.Xxx(...) — a known torch.nn module
  const nnMatch = callee.match(/(?:^|\.)nn\.(\w+)$/) || callee.match(/^nn\.(\w+)$/)
  if (nnMatch) {
    return { kind: "layer", layerType: nnMatch[1], args: expr.args, kwargs: expr.kwargs }
  }
  // bare `nn` attribute chain e.g. torch.nn.Conv2d
  const chain = calleeName(expr.func)
  if (chain && /\.nn\.\w+$/.test(chain)) {
    const lt = chain.split(".").pop()!
    return { kind: "layer", layerType: lt, args: expr.args, kwargs: expr.kwargs }
  }

  // Otherwise a user-defined submodule class (e.g. BasicBlock(...)).
  if (/^[A-Z]/.test(callee.split(".").pop() || "")) {
    return { kind: "custom", className: callee.split(".").pop()!, args: expr.args, kwargs: expr.kwargs }
  }
  return { kind: "unknown", label: callee }
}

// Return a dotted name for a callee expression, e.g. `nn.Conv2d`, `F.relu`,
// `torch.cat`, `x.view`. Returns null for non-name callees.
function calleeName(e: Expr): string | null {
  if (e.kind === "name") return e.id
  if (e.kind === "attr") {
    const base = calleeName(e.obj)
    return base ? `${base}.${e.attr}` : e.attr
  }
  return null
}

// ---------------------------------------------------------------------------
// Graph builder.
// ---------------------------------------------------------------------------
interface BuildContext {
  nodes: ParsedNode[]
  edges: ParsedEdge[]
  warnings: string[]
  unsupported: Set<string>
  classes: Map<string, ParsedClass>
  counter: { n: number }
}

// A tensor value flowing through forward(): the id of the node that produced it.
type TensorVal = { nodeId: string }

function newId(ctx: BuildContext, hint: string): string {
  ctx.counter.n += 1
  return `${hint}_${ctx.counter.n}`
}

function addNode(ctx: BuildContext, type: string, data: Record<string, any>, hint: string): string {
  const id = newId(ctx, hint)
  ctx.nodes.push({ id, type, data: { label: hint, ...data }, position: { x: 0, y: 0 } })
  return id
}

function connect(ctx: BuildContext, sources: (string | null | undefined)[], target: string, multi: boolean) {
  let handleIndex = 0
  for (const src of sources) {
    if (!src) continue
    handleIndex += 1
    const edge: ParsedEdge = {
      id: `e_${src}_${target}_${handleIndex}`,
      source: src,
      target,
      type: "default",
    }
    if (multi) edge.targetHandle = `input${handleIndex}`
    ctx.edges.push(edge)
  }
}

// Convert a layer descriptor's args/kwargs into node data.
function layerData(desc: { layerType: string; args: Expr[]; kwargs: Record<string, Expr> }): Record<string, any> {
  const data: Record<string, any> = {}
  for (const [k, v] of Object.entries(desc.kwargs)) {
    const val = exprToValue(v)
    if (val !== undefined) data[k] = val
  }
  const positional = POSITIONAL_ARGS[desc.layerType]
  if (positional) {
    desc.args.forEach((arg, idx) => {
      const key = positional[idx]
      if (key && data[key] === undefined) {
        const val = exprToValue(arg)
        if (val !== undefined) data[key] = val
      }
    })
  }
  return data
}

// Positional constructor argument order for common nn modules.
const POSITIONAL_ARGS: Record<string, string[]> = {
  Linear: ["in_features", "out_features", "bias"],
  LazyLinear: ["out_features", "bias"],
  Conv1d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
  Conv2d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
  Conv3d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias"],
  ConvTranspose1d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding"],
  ConvTranspose2d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding"],
  ConvTranspose3d: ["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding"],
  BatchNorm1d: ["num_features", "eps", "momentum"],
  BatchNorm2d: ["num_features", "eps", "momentum"],
  BatchNorm3d: ["num_features", "eps", "momentum"],
  InstanceNorm1d: ["num_features", "eps", "momentum"],
  InstanceNorm2d: ["num_features", "eps", "momentum"],
  InstanceNorm3d: ["num_features", "eps", "momentum"],
  LayerNorm: ["normalized_shape", "eps"],
  RMSNorm: ["normalized_shape", "eps"],
  GroupNorm: ["num_groups", "num_channels", "eps"],
  MaxPool1d: ["kernel_size", "stride", "padding"],
  MaxPool2d: ["kernel_size", "stride", "padding"],
  MaxPool3d: ["kernel_size", "stride", "padding"],
  AvgPool2d: ["kernel_size", "stride", "padding"],
  AdaptiveAvgPool2d: ["output_size"],
  AdaptiveMaxPool1d: ["output_size"],
  Dropout: ["p"],
  Dropout2d: ["p"],
  Dropout3d: ["p"],
  LeakyReLU: ["negative_slope", "inplace"],
  Softmax: ["dim"],
  LogSoftmax: ["dim"],
  Flatten: ["start_dim", "end_dim"],
  LSTM: ["input_size", "hidden_size", "num_layers"],
  GRU: ["input_size", "hidden_size", "num_layers"],
  RNN: ["input_size", "hidden_size", "num_layers"],
  MultiheadAttention: ["embed_dim", "num_heads", "dropout"],
  TransformerEncoderLayer: ["d_model", "nhead", "dim_feedforward", "dropout"],
  TransformerDecoderLayer: ["d_model", "nhead", "dim_feedforward", "dropout"],
  Embedding: ["num_embeddings", "embedding_dim"],
  Upsample: ["size", "scale_factor"],
}

// Instantiate the node(s) for a module descriptor and return input/output ids.
// `chainInputs` are the tensor node ids feeding this module. Returns the id of
// the module's output tensor (or null if it produced nothing).
function instantiateModule(
  ctx: BuildContext,
  desc: ModuleDescriptor,
  hint: string,
  inputs: (string | null)[],
  depth: number,
): string | null {
  switch (desc.kind) {
    case "layer": {
      const nodeType = NN_NODE_TYPE_MAP[desc.layerType]
      if (!nodeType) {
        ctx.unsupported.add(desc.layerType)
        ctx.warnings.push(`Unsupported layer '${desc.layerType}' — inserted as a pass-through node.`)
        const id = addNode(ctx, "defaultNode", { label: hint, moduleType: desc.layerType }, hint)
        connect(ctx, inputs, id, false)
        return id
      }
      const data = layerData(desc)
      const id = addNode(ctx, nodeType, data, hint)
      connect(ctx, inputs, id, MULTI_INPUT_NODE_TYPES.has(nodeType))
      return id
    }
    case "sequential": {
      let current = inputs
      let lastId: string | null = inputs[0] ?? null
      for (let i = 0; i < desc.items.length; i++) {
        const out = instantiateModule(ctx, desc.items[i], `${hint}_${i}`, current, depth)
        if (out) {
          lastId = out
          current = [out]
        }
      }
      return lastId
    }
    case "modulelist": {
      // A ModuleList isn't callable as a whole; individual indexing is handled
      // at the call site. If someone applies it directly, chain the items.
      let current = inputs
      let lastId: string | null = inputs[0] ?? null
      for (let i = 0; i < desc.items.length; i++) {
        const out = instantiateModule(ctx, desc.items[i], `${hint}_${i}`, current, depth)
        if (out) {
          lastId = out
          current = [out]
        }
      }
      return lastId
    }
    case "custom": {
      // Try to inline the submodule's own forward graph.
      const inlined = inlineCustomModule(ctx, desc.className, inputs, hint, depth + 1)
      if (inlined !== undefined) return inlined
      // Fallback: single opaque node labeled with the class name.
      ctx.unsupported.add(desc.className)
      const id = addNode(ctx, "defaultNode", { label: hint, moduleType: desc.className }, hint)
      connect(ctx, inputs, id, false)
      return id
    }
    case "unknown": {
      const id = addNode(ctx, "defaultNode", { label: hint, moduleType: desc.label }, hint)
      connect(ctx, inputs, id, false)
      return id
    }
  }
}

const MAX_INLINE_DEPTH = 6

// Recursively inline a user-defined submodule's forward() as a subgraph.
// Returns the output node id, or `undefined` if inlining isn't possible
// (so the caller can fall back to an opaque node).
function inlineCustomModule(
  ctx: BuildContext,
  className: string,
  inputs: (string | null)[],
  hint: string,
  depth: number,
): string | null | undefined {
  if (depth > MAX_INLINE_DEPTH) return undefined
  const cls = ctx.classes.get(className)
  if (!cls || !cls.forwardBody.trim() || cls.forwardArgs.length === 0) return undefined

  const scope = new Map<string, TensorVal>()
  // Bind the submodule's forward args to the incoming tensors.
  cls.forwardArgs.forEach((argName, i) => {
    const src = inputs[i] ?? inputs[0]
    if (src) scope.set(argName, { nodeId: src })
  })
  const result = buildForward(ctx, cls, scope, `${hint}`, depth)
  return result ?? undefined
}

// Walk a class's forward() statements, mutating ctx with nodes/edges.
// Returns the tensor node id produced by the (first) `return`.
function buildForward(
  ctx: BuildContext,
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hintPrefix: string,
  depth: number,
): string | null {
  let returnId: string | null = null
  for (const stmt of splitStatements(cls.forwardBody)) {
    // return statement
    const retMatch = stmt.match(/^return\s+([\s\S]+)$/)
    if (retMatch) {
      const expr = parseExpression(retMatch[1])
      if (expr) {
        const val = evalExpr(ctx, expr, cls, scope, hintPrefix, depth)
        returnId = val?.nodeId ?? returnId
      }
      break
    }
    if (/^\s*(if|for|while|with|else|elif|try|except|def|class|assert|raise|print|del)\b/.test(stmt)) {
      // Control flow / misc — best-effort: skip but keep any obvious call side effects.
      continue
    }
    // assignment: targets = expr
    const asgn = stmt.match(/^([A-Za-z_][\w\s,()]*?)\s*=\s*([\s\S]+)$/)
    if (asgn && !/[=!<>]=/.test(stmt.slice(0, stmt.indexOf("=") + 2).replace("==", ""))) {
      const targetsRaw = asgn[1].trim()
      const rhs = asgn[2].trim()
      const expr = parseExpression(rhs)
      if (!expr) continue
      const val = evalExpr(ctx, expr, cls, scope, hintPrefix, depth)
      // Handle tuple unpacking `out, _ = ...` — bind the first target to the value.
      const targets = targetsRaw
        .replace(/^\(|\)$/g, "")
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
      if (val) {
        if (targets.length <= 1) {
          scope.set(targets[0], val)
        } else {
          // Bind the first non-underscore target to the tensor value.
          const primary = targets.find((t) => t !== "_") ?? targets[0]
          scope.set(primary, val)
        }
      }
      continue
    }
    // bare expression statement (e.g. an in-place op) — evaluate for side effects.
    const expr = parseExpression(stmt)
    if (expr) evalExpr(ctx, expr, cls, scope, hintPrefix, depth)
  }
  return returnId
}

// Evaluate a forward-pass expression, creating nodes as needed, and return the
// tensor value it yields (or null for scalars / unresolved values).
function evalExpr(
  ctx: BuildContext,
  expr: Expr,
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hint: string,
  depth: number,
): TensorVal | null {
  switch (expr.kind) {
    case "name": {
      const v = scope.get(expr.id)
      return v ?? null
    }
    case "num":
    case "str":
    case "const":
      return null
    case "binop": {
      // Tensor arithmetic -> merge nodes. Skip if either side is a scalar.
      const left = evalExpr(ctx, expr.left, cls, scope, hint, depth)
      const right = evalExpr(ctx, expr.right, cls, scope, hint, depth)
      if (!left && !right) return null
      if (!left || !right) {
        // scalar-scaled tensor (e.g. `x * 0.5`) — pass the tensor through.
        return left ?? right
      }
      if (expr.op === "+" || expr.op === "-") {
        const id = addNode(ctx, "addNode", {}, "add")
        connect(ctx, [left.nodeId, right.nodeId], id, true)
        return { nodeId: id }
      }
      if (expr.op === "*" || expr.op === "@" || expr.op === "/") {
        const id = addNode(ctx, "multiplyNode", {}, "mul")
        connect(ctx, [left.nodeId, right.nodeId], id, true)
        return { nodeId: id }
      }
      return left
    }
    case "unary":
      return evalExpr(ctx, expr.operand, cls, scope, hint, depth)
    case "list":
    case "tuple": {
      // Used as call arguments (e.g. torch.cat([a, b])); return null here —
      // the call handler inspects elements directly.
      const vals = expr.elts.map((e) => evalExpr(ctx, e, cls, scope, hint, depth)).filter(Boolean)
      return (vals[0] as TensorVal) ?? null
    }
    case "index": {
      // e.g. self.experts[i] used as callee is handled in `call`; a plain index
      // like `x[0]` just follows the base tensor.
      return evalExpr(ctx, expr.obj, cls, scope, hint, depth)
    }
    case "attr": {
      // Attribute access that isn't a call (e.g. `x.shape`) yields no tensor.
      return evalExpr(ctx, expr.obj, cls, scope, hint, depth)
    }
    case "call":
      return evalCall(ctx, expr, cls, scope, hint, depth)
  }
}

function evalCall(
  ctx: BuildContext,
  expr: Expr & { kind: "call" },
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hint: string,
  depth: number,
): TensorVal | null {
  const func = expr.func

  // --- self.<module>(...) or self.<modulelist>[i](...) ---------------------
  const selfModule = resolveSelfModule(func, cls)
  if (selfModule) {
    const inputs = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
    const out = instantiateModule(ctx, selfModule.desc, selfModule.name, inputs, depth)
    return out ? { nodeId: out } : null
  }

  // --- functional / torch / tensor-method calls ----------------------------
  const name = calleeName(func)
  if (name) {
    const short = name.split(".").pop()!

    // torch.cat / torch.concat / torch.stack -> concatenate
    if (short === "cat" || short === "concat" || short === "stack") {
      const tensors = collectListTensors(ctx, expr.args, cls, scope, hint, depth)
      const id = addNode(ctx, "concatenateNode", { dim: kwargOrArgNumber(expr, "dim", 1) ?? 1 }, "concat")
      connect(ctx, tensors, id, true)
      return { nodeId: id }
    }
    // torch.add / tensor.add
    if (short === "add") {
      const ins = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
      const base = methodReceiver(ctx, func, cls, scope, hint, depth)
      const all = [base, ...ins].filter(Boolean) as string[]
      const id = addNode(ctx, "addNode", {}, "add")
      connect(ctx, all, id, true)
      return { nodeId: id }
    }
    // torch.mul / tensor.mul
    if (short === "mul" || short === "matmul" || short === "bmm") {
      const ins = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
      const base = methodReceiver(ctx, func, cls, scope, hint, depth)
      const all = [base, ...ins].filter(Boolean) as string[]
      const id = addNode(ctx, "multiplyNode", {}, "mul")
      connect(ctx, all, id, true)
      return { nodeId: id }
    }

    // functional activation / pool / dropout / flatten
    const fnType = FUNCTIONAL_NODE_TYPE_MAP[short]
    if (fnType && (name.startsWith("F.") || name.startsWith("torch.") || name.includes("functional.") || func.kind === "attr")) {
      // For a tensor method like `x.relu()`, the receiver is the input.
      const receiver = func.kind === "attr" ? methodReceiver(ctx, func, cls, scope, hint, depth) : null
      const argInputs = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
      const inputs = receiver ? [receiver, ...argInputs] : argInputs
      const data: Record<string, any> = {}
      const dim = kwargOrArgNumber(expr, "dim", -1)
      if (fnType === "softmaxNode" && dim !== null) data.dim = dim
      const id = addNode(ctx, fnType, data, short)
      connect(ctx, inputs.slice(0, 1), id, false)
      return { nodeId: id }
    }

    // reshape / view -> reshape or flatten
    if (short === "view" || short === "reshape") {
      const receiver = methodReceiver(ctx, func, cls, scope, hint, depth)
      const dims = expr.args.map(exprToValue).filter((v) => typeof v === "number")
      // A `.view(N, -1)` style flatten is common — detect 2D flatten.
      const isFlatten = expr.args.length === 2 && dims.includes(-1)
      const id = isFlatten
        ? addNode(ctx, "flattenNode", {}, "flatten")
        : addNode(ctx, "reshapeNode", { targetShape: dims.filter((d) => d !== -1) }, "reshape")
      connect(ctx, [receiver], id, false)
      return { nodeId: id }
    }
    if (short === "flatten") {
      const receiver = func.kind === "attr" ? methodReceiver(ctx, func, cls, scope, hint, depth) : null
      const argInputs = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
      const input = receiver ?? argInputs[0] ?? null
      const id = addNode(ctx, "flattenNode", {}, "flatten")
      connect(ctx, [input], id, false)
      return { nodeId: id }
    }
    if (short === "transpose" || short === "permute") {
      const receiver = func.kind === "attr" ? methodReceiver(ctx, func, cls, scope, hint, depth) : null
      const argInputs = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
      const input = receiver ?? argInputs[0] ?? null
      const id = addNode(ctx, "transposeNode", {}, "transpose")
      connect(ctx, [input], id, false)
      return { nodeId: id }
    }

    // Pass-through tensor methods that don't change the graph.
    if (["contiguous", "to", "type", "float", "double", "half", "cuda", "cpu", "clone", "detach", "squeeze", "unsqueeze"].includes(short)) {
      return func.kind === "attr" ? evalExpr(ctx, (func as any).obj, cls, scope, hint, depth) : null
    }

    // Aggregations that reduce dimensions — treat as pass-through tensors.
    if (["mean", "sum", "max", "min", "amax", "amin"].includes(short) && func.kind === "attr") {
      return evalExpr(ctx, (func as any).obj, cls, scope, hint, depth)
    }

    // Non-tensor helpers (size, shape, dim, len, range, int, ...) -> scalar.
    if (["size", "shape", "dim", "numel", "len", "int", "float32", "range", "type_as"].includes(short)) {
      return null
    }
  }

  // Unknown call — try to pass through the first tensor argument so the graph
  // stays connected.
  const ins = collectTensorInputs(ctx, expr.args, cls, scope, hint, depth)
  if (ins.length > 0 && ins[0]) return { nodeId: ins[0] }
  return null
}

// If `func` refers to `self.<name>` or `self.<name>[idx]`, resolve to the
// module descriptor and a display name.
function resolveSelfModule(
  func: Expr,
  cls: ParsedClass,
): { desc: ModuleDescriptor; name: string } | null {
  // self.name
  if (func.kind === "attr" && func.obj.kind === "name" && func.obj.id === "self") {
    const desc = cls.modules.get(func.attr)
    if (desc) return { desc, name: func.attr }
    return null
  }
  // self.name[idx]
  if (func.kind === "index" && func.obj.kind === "attr" && func.obj.obj.kind === "name" && func.obj.obj.id === "self") {
    const listName = func.obj.attr
    const listDesc = cls.modules.get(listName)
    if (listDesc && listDesc.kind === "modulelist") {
      let idx = 0
      if (func.index && func.index.kind === "num") idx = func.index.value
      const item = listDesc.items[idx] ?? listDesc.items[0]
      if (item) return { desc: item, name: `${listName}_${idx}` }
    }
  }
  return null
}

// The receiver tensor of a method call `x.method(...)` -> x's node id.
function methodReceiver(
  ctx: BuildContext,
  func: Expr,
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hint: string,
  depth: number,
): string | null {
  if (func.kind === "attr") {
    const v = evalExpr(ctx, func.obj, cls, scope, hint, depth)
    return v?.nodeId ?? null
  }
  return null
}

// Collect tensor node ids from a list of argument expressions (flattening any
// list/tuple argument, e.g. torch.cat([a, b])).
function collectTensorInputs(
  ctx: BuildContext,
  args: Expr[],
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hint: string,
  depth: number,
): string[] {
  const ids: string[] = []
  for (const a of args) {
    if (a.kind === "list" || a.kind === "tuple") {
      for (const el of a.elts) {
        const v = evalExpr(ctx, el, cls, scope, hint, depth)
        if (v) ids.push(v.nodeId)
      }
    } else {
      const v = evalExpr(ctx, a, cls, scope, hint, depth)
      if (v) ids.push(v.nodeId)
    }
  }
  return ids
}

// Like collectTensorInputs but specifically for the list argument of cat/stack.
function collectListTensors(
  ctx: BuildContext,
  args: Expr[],
  cls: ParsedClass,
  scope: Map<string, TensorVal>,
  hint: string,
  depth: number,
): string[] {
  const first = args[0]
  if (first && (first.kind === "list" || first.kind === "tuple")) {
    const ids: string[] = []
    for (const el of first.elts) {
      const v = evalExpr(ctx, el, cls, scope, hint, depth)
      if (v) ids.push(v.nodeId)
    }
    return ids
  }
  return collectTensorInputs(ctx, args, cls, scope, hint, depth)
}

function kwargOrArgNumber(expr: Expr & { kind: "call" }, kw: string, argIndex: number): number | null {
  if (expr.kwargs[kw] !== undefined) {
    const v = exprToValue(expr.kwargs[kw])
    return typeof v === "number" ? v : null
  }
  return null
}

// ---------------------------------------------------------------------------
// Layout: layered top-to-bottom placement using longest-path depth.
// ---------------------------------------------------------------------------
function layoutGraph(nodes: ParsedNode[], edges: ParsedEdge[]) {
  const idToNode = new Map(nodes.map((n) => [n.id, n]))
  const incoming = new Map<string, string[]>()
  const outgoing = new Map<string, string[]>()
  for (const n of nodes) {
    incoming.set(n.id, [])
    outgoing.set(n.id, [])
  }
  for (const e of edges) {
    if (idToNode.has(e.source) && idToNode.has(e.target)) {
      outgoing.get(e.source)!.push(e.target)
      incoming.get(e.target)!.push(e.source)
    }
  }
  // Longest-path depth via memoized DFS (graph is a DAG in the common case).
  const depthCache = new Map<string, number>()
  const visiting = new Set<string>()
  function depthOf(id: string): number {
    if (depthCache.has(id)) return depthCache.get(id)!
    if (visiting.has(id)) return 0 // cycle guard
    visiting.add(id)
    const preds = incoming.get(id) || []
    let d = 0
    for (const p of preds) d = Math.max(d, depthOf(p) + 1)
    visiting.delete(id)
    depthCache.set(id, d)
    return d
  }
  const byDepth = new Map<number, string[]>()
  for (const n of nodes) {
    const d = depthOf(n.id)
    if (!byDepth.has(d)) byDepth.set(d, [])
    byDepth.get(d)!.push(n.id)
  }
  const X_SPACING = 240
  const Y_SPACING = 130
  const X_ORIGIN = 300
  for (const [d, ids] of byDepth) {
    const count = ids.length
    ids.forEach((id, i) => {
      const node = idToNode.get(id)!
      const offset = (i - (count - 1) / 2) * X_SPACING
      node.position = { x: X_ORIGIN + offset, y: 40 + d * Y_SPACING }
    })
  }
}

// ---------------------------------------------------------------------------
// Main entry point.
// ---------------------------------------------------------------------------
export function parsePyTorchModel(code: string): ParseResult {
  const errors: string[] = []
  const warnings: string[] = []

  if (!code || !code.trim()) {
    return { nodes: [], edges: [], errors: ["No code provided."], warnings: [], unsupportedModules: [] }
  }

  const cleaned = stripComments(code)
  const classes = extractClasses(cleaned)

  if (classes.size === 0) {
    return {
      nodes: [],
      edges: [],
      errors: ["No PyTorch model class found. Expected 'class YourModel(nn.Module):'."],
      warnings: [],
      unsupportedModules: [],
    }
  }

  // Pick the main class: prefer one that other classes don't instantiate as a
  // submodule (the top-level model), otherwise the last defined class.
  const mainClass = selectMainClass(classes)
  if (!mainClass) {
    return {
      nodes: [],
      edges: [],
      errors: ["Could not identify a model class with a forward() method."],
      warnings: [],
      unsupportedModules: [],
    }
  }

  const ctx: BuildContext = {
    nodes: [],
    edges: [],
    warnings,
    unsupported: new Set<string>(),
    classes,
    counter: { n: 0 },
  }

  // Create the input node(s) and seed the scope.
  const scope = new Map<string, TensorVal>()
  const inputArgs = mainClass.forwardArgs.length > 0 ? mainClass.forwardArgs : ["x"]
  // Only the first argument is treated as the primary tensor input for layout;
  // additional args also become input nodes so multi-input models parse.
  for (const argName of inputArgs) {
    const inputId = addNode(ctx, "inputNode", { name: argName, channels: 3, height: 224, width: 224 }, argName)
    scope.set(argName, { nodeId: inputId })
  }

  let outputId: string | null = null
  if (mainClass.forwardBody.trim()) {
    outputId = buildForward(ctx, mainClass, scope, "m", 0)
  } else {
    // No forward(): fall back to chaining the declared modules linearly.
    warnings.push("No forward() method found — layers were connected sequentially in declaration order.")
    let current: string | null = ctx.nodes[0]?.id ?? null
    for (const [name, desc] of mainClass.modules) {
      const out = instantiateModule(ctx, desc, name, [current], 0)
      if (out) current = out
    }
    outputId = current
  }

  // Append an output node for the returned tensor.
  if (outputId) {
    const outId = addNode(ctx, "outputNode", { label: "output" }, "output")
    connect(ctx, [outputId], outId, false)
  }

  if (ctx.nodes.filter((n) => n.type !== "inputNode" && n.type !== "outputNode").length === 0) {
    errors.push(
      "No layers or operations could be extracted from the model. Ensure the forward() method uses self.<layer>(...) calls or supported functional ops.",
    )
  }

  layoutGraph(ctx.nodes, ctx.edges)

  const unsupportedModules = [...ctx.unsupported]
  if (unsupportedModules.length > 0) {
    warnings.push(`Unsupported module types kept as pass-through nodes: ${unsupportedModules.join(", ")}.`)
  }

  return { nodes: ctx.nodes, edges: ctx.edges, errors, warnings, unsupportedModules }
}

function selectMainClass(classes: Map<string, ParsedClass>): ParsedClass | null {
  const withForward = [...classes.values()].filter((c) => c.forwardBody.trim() && c.forwardArgs.length > 0)
  if (withForward.length === 0) {
    // Maybe a class with __init__ modules but no forward — still usable.
    const withModules = [...classes.values()].filter((c) => c.modules.size > 0)
    return withModules[withModules.length - 1] ?? null
  }
  if (withForward.length === 1) return withForward[0]

  // Count references: a class instantiated inside another class's __init__ is a
  // submodule, not the top-level model.
  const referenced = new Set<string>()
  for (const c of classes.values()) {
    for (const desc of c.modules.values()) {
      collectCustomClassNames(desc, referenced)
    }
  }
  const topLevel = withForward.filter((c) => !referenced.has(c.name))
  if (topLevel.length === 1) return topLevel[0]
  if (topLevel.length > 1) return topLevel[topLevel.length - 1]
  return withForward[withForward.length - 1]
}

function collectCustomClassNames(desc: ModuleDescriptor, out: Set<string>) {
  if (desc.kind === "custom") out.add(desc.className)
  else if (desc.kind === "sequential" || desc.kind === "modulelist") {
    for (const item of desc.items) collectCustomClassNames(item, out)
  }
}

// Convenience: adapt the parser output to the app's GraphNode/GraphEdge shape.
export function toGraphIR(result: ParseResult): { nodes: GraphNode[]; edges: GraphEdge[] } {
  return {
    nodes: result.nodes.map((n) => ({ id: n.id, type: n.type, data: n.data as any, position: n.position })),
    edges: result.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: e.type,
      targetHandle: e.targetHandle,
    })),
  }
}
