import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"

interface FractionalMaxPool2DNodeProps {
  data: {
    kernel_size?: number
    output_ratio?: number
    inputShape?: { batch: number; channels: number; height: number; width: number }
    outputShape?: { batch: number; channels: number; height: number; width: number }
  }
}

export function FractionalMaxPool2DNode({ data }: FractionalMaxPool2DNodeProps) {
  return (
    <Card className="min-w-[180px] bg-white border-2 border-red-200 shadow-sm">
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      <div className="p-3">
        <div className="font-semibold text-sm text-gray-800 mb-2">FractionalMaxPool2D</div>
        <div className="text-xs text-gray-600 space-y-1">
          <div>Kernel: {data.kernel_size || 2}</div>
          <div>Output Ratio: {data.output_ratio || 0.5}</div>
          {data.inputShape && (
            <div className="text-blue-600">
              In: [{data.inputShape.batch}, {data.inputShape.channels}, {data.inputShape.height},{" "}
              {data.inputShape.width}]
            </div>
          )}
          {data.outputShape && (
            <div className="text-green-600">
              Out: [{data.outputShape.batch}, {data.outputShape.channels}, {data.outputShape.height},{" "}
              {data.outputShape.width}]
            </div>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </Card>
  )
}
