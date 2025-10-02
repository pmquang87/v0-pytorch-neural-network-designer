import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface ScaledDotProductAttentionNodeProps {
  data: NodeData
}

export function ScaledDotProductAttentionNode({ data }: ScaledDotProductAttentionNodeProps) {
  return (
    <div className="bg-blue-100 border-2 border-blue-300 rounded-lg p-3 min-w-[160px]">
      <Handle type="target" position={Position.Left} id="query" style={{ top: "25%" }} />
      <Handle type="target" position={Position.Left} id="key" style={{ top: "50%" }} />
      <Handle type="target" position={Position.Left} id="value" style={{ top: "75%" }} />
      <div className="text-center">
        <div className="font-semibold text-blue-800">Scaled Dot-Product Attention</div>
        <div className="text-xs text-blue-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-blue-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
