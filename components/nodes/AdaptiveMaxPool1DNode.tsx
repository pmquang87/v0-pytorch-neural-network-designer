import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"

interface AdaptiveMaxPool1DNodeProps {
  data: {
    output_size?: number
    inputShape?: { batch: number; channels: number; length: number }
    outputShape?: { batch: number; channels: number; length: number }
  }
}

export function AdaptiveMaxPool1DNode({ data }: AdaptiveMaxPool1DNodeProps) {
  return (
    <Card className="min-w-[180px] bg-white border-2 border-red-200 shadow-sm">
      <Handle type="target" position={Position.Top} className="w-3 h-3" />

      <div className="p-3">
        <div className="font-semibold text-sm text-gray-800 mb-2">AdaptiveMaxPool1D</div>
        <div className="text-xs text-gray-600 space-y-1">
          <div>Output Size: {data.output_size || 1}</div>
          {data.inputShape && (
            <div className="text-blue-600">
              In: [{data.inputShape.batch}, {data.inputShape.channels}, {data.inputShape.length}]
            </div>
          )}
          {data.outputShape && (
            <div className="text-green-600">
              Out: [{data.outputShape.batch}, {data.outputShape.channels}, {data.outputShape.length}]
            </div>
          )}
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} className="w-3 h-3" />
    </Card>
  )
}
