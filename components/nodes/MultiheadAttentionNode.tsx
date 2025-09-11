import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface MultiheadAttentionNodeProps {
  data: NodeData
}

export function MultiheadAttentionNode({ data }: MultiheadAttentionNodeProps) {
  return (
    <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-3 min-w-[160px]">
      <Handle type="target" position={Position.Left} id="query" style={{ top: "25%" }} />
      <Handle type="target" position={Position.Left} id="key" style={{ top: "50%" }} />
      <Handle type="target" position={Position.Left} id="value" style={{ top: "75%" }} />
      <div className="text-center">
        <div className="font-semibold text-purple-800">MultiheadAttention</div>
        <div className="text-xs text-purple-600 mt-1">Embed: {data.embed_dim || 512}</div>
        <div className="text-xs text-purple-600">Heads: {data.num_heads || 8}</div>
        <div className="text-xs text-purple-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-purple-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
