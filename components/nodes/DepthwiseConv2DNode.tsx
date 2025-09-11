import { Handle, Position } from "@xyflow/react"
import type { NodeProps } from "@xyflow/react"

export function DepthwiseConv2DNode({ data }: NodeProps) {
  return (
    <div className="bg-orange-100 border-2 border-orange-300 rounded-lg p-3 min-w-[180px]">
      <Handle type="target" position={Position.Top} />

      <div className="text-center">
        <div className="font-semibold text-orange-800 mb-1">DepthwiseConv2D</div>
        <div className="text-xs text-orange-600 space-y-1">
          <div>Channels: {data.in_channels || 32}</div>
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
