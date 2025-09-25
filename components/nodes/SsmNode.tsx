import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { BrainCircuit } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function SsmNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, channels: 1, features: 512 }
  const outputShape = calculateOutputShape("ssmNode", [inputShape], data)

  return (
    <Card className="min-w-[200px] border-2 border-orange-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-orange-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <BrainCircuit className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">SSM (Mamba)</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>D Model: {data.d_model || "?"}</div>
          <div>D State: {data.d_state || "?"}</div>
          <div>D Conv: {data.d_conv || "?"}</div>
          <div>Expand: {data.expand || "?"}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500 border-2 border-background" />
    </Card>
  )
}
