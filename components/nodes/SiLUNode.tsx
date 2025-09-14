import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface SiLUNodeProps {
  data: NodeData
}

export function SiLUNode({ data }: SiLUNodeProps) {
  return (
    <div className="bg-orange-100 border-2 border-orange-300 rounded-lg p-3 min-w-[120px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-orange-800">SiLU</div>
        <div className="text-xs text-orange-600 mt-1">
          In: {formatTensorShape(data.inputShape)}
        </div>
        <div className="text-xs text-orange-600">
          Out: {formatTensorShape(data.outputShape)}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
