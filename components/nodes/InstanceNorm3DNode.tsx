import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface InstanceNorm3DNodeProps {
  data: NodeData
}

export function InstanceNorm3DNode({ data }: InstanceNorm3DNodeProps) {
  return (
    <div className="bg-cyan-100 border-2 border-cyan-300 rounded-lg p-3 min-w-[140px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-cyan-800">InstanceNorm3D</div>
        <div className="text-xs text-cyan-600 mt-1">Features: {data.num_features || 16}</div>
        <div className="text-xs text-cyan-600 mt-1">
          In: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-cyan-600">
          Out: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
