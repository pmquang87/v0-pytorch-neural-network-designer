import { Handle, Position } from "@xyflow/react"
import type { NodeData } from "@/lib/types"

interface InvertedResidualBlockNodeProps {
  data: NodeData
}

export function InvertedResidualBlockNode({ data }: InvertedResidualBlockNodeProps) {
  return (
    <div className="bg-blue-100 border-2 border-blue-300 rounded-lg p-3 min-w-[160px]">
      <Handle type="target" position={Position.Left} />
      <div className="text-center">
        <div className="font-semibold text-blue-800">InvertedResidualBlock</div>
        <div className="text-xs text-blue-600 mt-1">In: {data.in_channels || "?"}</div>
        <div className="text-xs text-blue-600">Out: {data.out_channels || "?"}</div>
        <div className="text-xs text-blue-600">Stride: {data.stride || "?"}</div>
        <div className="text-xs text-blue-600">Expand: {data.expand_ratio || "?"}</div>
        <div className="text-xs text-blue-600 mt-1">
          In Shape: {data.inputShape ? `[${Object.values(data.inputShape).join(",")}]` : "[?]"}
        </div>
        <div className="text-xs text-blue-600">
          Out Shape: {data.outputShape ? `[${Object.values(data.outputShape).join(",")}]` : "[?]"}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
