import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface GELUNodeProps {
  data: NodeData
}

export function GELUNode({ data }: GELUNodeProps) {
  return (
    <div className="bg-orange-100 border-2 border-orange-300 rounded-lg p-3 min-w-[120px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-orange-800">GELU</div>
        <div className="text-xs text-orange-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-orange-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
