import { memo } from "react"
import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface InstanceNorm1DNodeProps {
  data: NodeData
}

function InstanceNorm1DNodeImpl({ data }: InstanceNorm1DNodeProps) {
  return (
    <div className="bg-cyan-100 border-2 border-cyan-300 rounded-lg p-3 min-w-[140px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-cyan-800">InstanceNorm1D</div>
        <div className="text-xs text-cyan-600 mt-1">Features: {data.num_features || 128}</div>
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

export const InstanceNorm1DNode = memo(InstanceNorm1DNodeImpl)
InstanceNorm1DNode.displayName = "InstanceNorm1DNode"
