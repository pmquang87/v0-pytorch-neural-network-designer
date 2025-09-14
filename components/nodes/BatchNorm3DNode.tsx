import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { BarChart3 } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"
import type { NodeData } from "@/lib/types"

export function BatchNorm3DNode({ data }: { data: NodeData }) {
  return (
    <Card className="min-w-[150px] border-2 border-violet-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-violet-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="h-4 w-4 text-violet-500" />
          <span className="font-medium text-sm">BatchNorm3D</span>
        </div>
        <div className="text-xs text-muted-foreground">Features: {data.num_features || "?"}</div>
        <div className="text-xs text-muted-foreground mt-1">
          In: {formatTensorShape(data.inputShape)}
        </div>
        <div className="text-xs text-muted-foreground">
          Out: {formatTensorShape(data.outputShape)}
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-violet-500 border-2 border-background" />
    </Card>
  )
}
