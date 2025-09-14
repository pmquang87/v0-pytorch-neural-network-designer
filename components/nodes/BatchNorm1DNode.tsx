import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { BarChart3 } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"
import type { NodeData } from "@/lib/types"

export function BatchNorm1DNode({ data }: { data: NodeData }) {
  return (
    <Card className="min-w-[160px] bg-card border-2 border-purple-500/50 shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-purple-500 border-2 border-background" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="h-4 w-4 text-purple-500" />
          <span className="font-medium text-sm">BatchNorm1D</span>
        </div>
        <div className="text-xs text-muted-foreground">features: {data.num_features ?? 128}</div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(data.inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(data.outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-purple-500 border-2 border-background" />
    </Card>
  )
}
