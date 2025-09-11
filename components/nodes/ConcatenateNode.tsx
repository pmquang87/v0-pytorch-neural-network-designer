import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Merge } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function ConcatenateNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, features: 128 }
  const outputShape = calculateOutputShape("concatenateNode", inputShape, data)
  const numInputs = data.num_inputs || 2

  const inputHandles = []
  for (let i = 1; i <= numInputs; i++) {
    const topPercentage = (i / (numInputs + 1)) * 100
    inputHandles.push(
      <Handle
        key={`input${i}`}
        type="target"
        position={Position.Left}
        id={`input${i}`}
        style={{ top: `${topPercentage}%` }}
        className="w-3 h-3 bg-green-500 border-2 border-background"
      />,
    )
  }

  return (
    <Card className="min-w-[160px] bg-card border-2 border-green-500/50 shadow-sm">
      {inputHandles}

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Merge className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Concatenate</span>
        </div>
        <div className="text-xs text-muted-foreground">dim: {data.dim ?? 1}</div>
        <div className="text-xs text-muted-foreground">inputs: {numInputs}</div>
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
