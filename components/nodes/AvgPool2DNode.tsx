import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Shrink } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function AvgPool2DNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, channels: 32, height: 28, width: 28 }
  const outputShape = calculateOutputShape("avgpool2dNode", [inputShape], data)

  return (
    <Card className="min-w-[160px] bg-card border-2 border-teal-500/50 shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-teal-500 border-2 border-background" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Shrink className="h-4 w-4 text-teal-500" />
          <span className="font-medium text-sm">AvgPool2D</span>
        </div>
        <div className="text-xs text-muted-foreground">
          kernel: {data.kernel_size ?? 2}
          <br />
          stride: {data.stride ?? 2}
          <br />
          padding: {data.padding ?? 0}
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-teal-500 border-2 border-background" />
    </Card>
  )
}
