import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Layers } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function LinearNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, features: 128 }
  const outputShape = calculateOutputShape("linearNode", [inputShape], data)

  return (
    <Card className="min-w-[150px] border-2 border-blue-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-blue-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-blue-500" />
          <span className="font-medium text-sm">Linear</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>In: {data.in_features || "?"}</div>
          <div>Out: {data.out_features || "?"}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-blue-500 border-2 border-background" />
    </Card>
  )
}
