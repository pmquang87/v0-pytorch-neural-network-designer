import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Plus } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function AddNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, features: 128 }
  const outputShape = calculateOutputShape("addNode", inputShape, data)

  const numInputs = data.num_inputs || 2

  return (
    <Card className="min-w-[160px] bg-card border-orange-500 border-2 shadow-sm">
      {Array.from({ length: numInputs }, (_, i) => (
        <Handle
          key={`input${i + 1}`}
          type="target"
          position={Position.Left}
          id={`input${i + 1}`}
          style={{ top: `${(100 / (numInputs + 1)) * (i + 1)}%` }}
          className="w-3 h-3 bg-orange-500"
        />
      ))}

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Plus className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">Add</span>
        </div>
        <div className="text-xs text-muted-foreground">{numInputs} inputs • Element-wise addition</div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500" />
    </Card>
  )
}
