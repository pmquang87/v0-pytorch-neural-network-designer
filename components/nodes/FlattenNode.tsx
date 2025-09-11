import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Minimize2 } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function FlattenNode({ data }: { data: any }) {
  const inputShape = data.inputShape || { batch: 1, channels: 1, height: 28, width: 28 }
  const outputShape = data.outputShape || { batch: 1, features: 784 }

  return (
    <Card className="min-w-[120px] border-2 border-cyan-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-cyan-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Minimize2 className="h-4 w-4 text-cyan-500" />
          <span className="font-medium text-sm">Flatten</span>
        </div>
        <div className="text-xs space-y-1">
          <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
          <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-cyan-500 border-2 border-background" />
    </Card>
  )
}
