import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface TransformerDecoderLayerNodeProps {
  data: NodeData
}

export function TransformerDecoderLayerNode({ data }: TransformerDecoderLayerNodeProps) {
  return (
    <div className="bg-violet-100 border-2 border-violet-300 rounded-lg p-3 min-w-[180px]">
      <Handle type="target" position={Position.Left} id="tgt" style={{ top: "33%" }} />
      <Handle type="target" position={Position.Left} id="memory" style={{ top: "66%" }} />
      <div className="text-center">
        <div className="font-semibold text-violet-800">TransformerDecoder</div>
        <div className="text-xs text-violet-600 mt-1">d_model: {data.d_model || 512}</div>
        <div className="text-xs text-violet-600">nhead: {data.nhead || 8}</div>
        <div className="text-xs text-violet-600">dim_ff: {data.dim_feedforward || 2048}</div>
        <div className="text-xs text-violet-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-violet-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
