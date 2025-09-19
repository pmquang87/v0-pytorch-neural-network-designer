import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Sigma } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function SigmoidNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-2 border-orange-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-orange-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Sigma className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">Sigmoid</span>
        </div>
        <div className="text-xs space-y-1">
          <div className="text-green-600">In: {formatTensorShape(data.inputShape)}</div>
          <div className="text-blue-600">Out: {formatTensorShape(data.outputShape)}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500 border-2 border-background" />
    </Card>
  )
}
