import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Plus } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function AddNode({ data }: { data: any }) {
  const numInputs = data.num_inputs || 2

  // data.inputShape is an array of shapes (or undefined) populated by propagateTensorShapes
  const inputShapes: (TensorShape | undefined)[] = Array.isArray(data.inputShape) ? data.inputShape : []

  // Find the first valid input shape for display
  const firstValidInputShape = inputShapes.find(s => s && Object.keys(s).length > 0)

  // Get all defined input shapes for output calculation
  const definedInputShapes = inputShapes.filter((s): s is TensorShape => s && Object.keys(s).length > 0)

  // Calculate output shape using the tensor shape calculator
  const outputShape = calculateOutputShape("addNode", definedInputShapes, data)

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
        className="w-3 h-3 bg-orange-500 border-2 border-background"
      />,
    )
  }

  return (
    <Card className="min-w-[160px] bg-card border-2 border-orange-500/50 shadow-sm">
      {inputHandles}

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Plus className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">Add</span>
        </div>
        <div className="text-xs text-muted-foreground">inputs: {numInputs}</div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-orange-600">
              In: {formatTensorShape(firstValidInputShape || {})}
            </div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500 border-2 border-background" />
    </Card>
  )
}
