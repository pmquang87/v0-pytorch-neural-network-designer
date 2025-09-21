import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { LogOut } from "lucide-react"
import { formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function OutputNode({ data }: { data: any }) {
  // For output nodes, we typically show the incoming tensor shape
  const inputShape: TensorShape | undefined = Array.isArray(data.inputShape)
    ? data.inputShape[0]
    : (data.inputShape as TensorShape | undefined)

  const outputShape: TensorShape | undefined = (data.outputShape as TensorShape | undefined) || inputShape

  return (
    <Card className="min-w-[160px] bg-card border-red-500/50 shadow-sm">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <LogOut className="h-4 w-4 text-red-500" />
          <span className="font-medium text-sm">Output</span>
        </div>
        <div className="text-xs text-muted-foreground">
          <div className="text-orange-600">In: {formatTensorShape(inputShape || {})}</div>
          <div className="text-blue-600">Out: {formatTensorShape(outputShape || {})}</div>
        </div>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        className="w-3 h-3 bg-red-500 border-2 border-background"
      />
    </Card>
  )
}
