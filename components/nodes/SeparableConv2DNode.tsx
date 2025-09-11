import { Handle, Position } from "@xyflow/react"
import type { NodeProps } from "@xyflow/react"

export function SeparableConv2DNode({ data }: NodeProps) {
  return (
    <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-3 min-w-[180px]">
      <Handle type="target" position={Position.Top} />

      <div className="text-center">
        <div className="font-semibold text-purple-800 mb-1">SeparableConv2D</div>
        <div className="text-xs text-purple-600 space-y-1">
          <div>
            In: {data.in_channels || 32} → Out: {data.out_channels || 64}
          </div>
          <div>Kernel: {data.kernel_size || 3}</div>
          <div>Stride: {data.stride || 1}</div>
          <div>Padding: {data.padding || 1}</div>
          {data.inputShape && (
            <div className="text-blue-600">
              In: [{data.inputShape.batch},{data.inputShape.channels},{data.inputShape.height},{data.inputShape.width}]
            </div>
          )}
          {data.outputShape && (
            <div className="text-green-600">
              Out: [{data.outputShape.batch},{data.outputShape.channels},{data.outputShape.height},
              {data.outputShape.width}]
            </div>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}
