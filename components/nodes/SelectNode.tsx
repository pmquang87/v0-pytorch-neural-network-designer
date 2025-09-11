import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface SelectNodeData extends NodeData {
  dim?: number
  index?: number
}

interface SelectNodeProps {
  data: SelectNodeData
}

export function SelectNode({ data }: SelectNodeProps) {
  return (
    <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-3 min-w-[120px]">
      <Handle type="target" position={Position.Left} />

      <div className="text-center">
        <div className="font-semibold text-purple-800 text-sm mb-1">Select</div>
        <div className="text-xs text-purple-600">
          dim: {data.dim ?? 0}, idx: {data.index ?? 0}
        </div>
        {data.inputShape && <div className="text-xs text-gray-500 mt-1">In: {JSON.stringify(data.inputShape)}</div>}
        {data.outputShape && <div className="text-xs text-gray-500">Out: {JSON.stringify(data.outputShape)}</div>}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
}
