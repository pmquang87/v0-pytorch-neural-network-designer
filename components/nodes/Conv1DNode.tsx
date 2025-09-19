import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Layers } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function Conv1DNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, channels: 1, length: 100 }
  const outputShape = calculateOutputShape("conv1dNode", [inputShape], data)

  return (
    <Card className="min-w-[150px] border-2 border-green-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-green-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Conv1D</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>In: {data.in_channels || 1}</div>
          <div>Out: {data.out_channels || 32}</div>
          <div>Kernel: {data.kernel_size || 3}</div>
          <div>Stride: {data.stride || 1}</div>
          <div>Padding: {data.padding || 0}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-green-500 border-2 border-background" />
    </Card>
  )
}
