import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Waves } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function TanhNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[120px] border-2 border-yellow-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-yellow-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Waves className="h-4 w-4 text-yellow-500" />
          <span className="font-medium text-sm">Tanh</span>
        </div>
        <div className="text-xs space-y-1">
          <div className="text-green-600">In: {formatTensorShape(data.inputShape)}</div>
          <div className="text-blue-600">Out: {formatTensorShape(data.outputShape)}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-yellow-500 border-2 border-background" />
    </Card>
  )
}
