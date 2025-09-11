import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface TransformerEncoderLayerNodeProps {
  data: NodeData
}

export function TransformerEncoderLayerNode({ data }: TransformerEncoderLayerNodeProps) {
  return (
    <div className="bg-indigo-100 border-2 border-indigo-300 rounded-lg p-3 min-w-[180px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-indigo-800">TransformerEncoder</div>
        <div className="text-xs text-indigo-600 mt-1">d_model: {data.d_model || 512}</div>
        <div className="text-xs text-indigo-600">nhead: {data.nhead || 8}</div>
        <div className="text-xs text-indigo-600">dim_ff: {data.dim_feedforward || 2048}</div>
        <div className="text-xs text-indigo-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-indigo-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
