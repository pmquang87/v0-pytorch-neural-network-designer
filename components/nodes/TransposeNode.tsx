import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface TransposeNodeData extends NodeData {
  dim0?: number
  dim1?: number
}

interface TransposeNodeProps {
  data: TransposeNodeData
}

export function TransposeNode({ data }: TransposeNodeProps) {
  return (
    <div className="bg-orange-100 border-2 border-orange-300 rounded-lg p-3 min-w-[120px]">
      <Handle type="target" position={Position.Left} />

      <div className="text-center">
        <div className="font-semibold text-orange-800 text-sm mb-1">Transpose</div>
        <div className="text-xs text-orange-600">
          dim0: {data.dim0 ?? 0}, dim1: {data.dim1 ?? 1}
        </div>
        {data.inputShape && <div className="text-xs text-gray-500 mt-1">In: {formatTensorShape(data.inputShape)}</div>}
        {data.outputShape && <div className="text-xs text-gray-500">Out: {formatTensorShape(data.outputShape)}</div>}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
}
