import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface InstanceNorm2DNodeProps {
  data: NodeData
}

export function InstanceNorm2DNode({ data }: InstanceNorm2DNodeProps) {
  return (
    <div className="bg-cyan-100 border-2 border-cyan-300 rounded-lg p-3 min-w-[140px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-cyan-800">InstanceNorm2D</div>
        <div className="text-xs text-cyan-600 mt-1">Features: {data.num_features || 32}</div>
        <div className="text-xs text-cyan-600 mt-1">
          In: {formatTensorShape(data.inputShape)}
        </div>
        <div className="text-xs text-cyan-600">
          Out: {formatTensorShape(data.outputShape)}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
