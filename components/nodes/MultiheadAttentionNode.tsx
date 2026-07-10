import { memo } from "react"
import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface MultiheadAttentionNodeProps {
  data: NodeData
}

function MultiheadAttentionNodeImpl({ data }: MultiheadAttentionNodeProps) {
  return (
    <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-3 min-w-[160px]">
      <Handle type="target" position={Position.Left} id="query" style={{ top: "25%" }} />
      <Handle type="target" position={Position.Left} id="key" style={{ top: "50%" }} />
      <Handle type="target" position={Position.Left} id="value" style={{ top: "75%" }} />
      <div className="text-center">
        <div className="font-semibold text-purple-800">MultiheadAttention</div>
        <div className="text-xs text-purple-600 mt-1">Embed: {data.embed_dim || 128}</div>
        <div className="text-xs text-purple-600">Heads: {data.num_heads || 8}</div>
        <div className="text-xs text-purple-600 mt-1">In: {formatTensorShape(data.inputShape)}</div>
        <div className="text-xs text-purple-600">Out: {formatTensorShape(data.outputShape)}</div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

export const MultiheadAttentionNode = memo(MultiheadAttentionNodeImpl)
MultiheadAttentionNode.displayName = "MultiheadAttentionNode"
