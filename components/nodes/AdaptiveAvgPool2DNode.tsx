import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Shrink } from "lucide-react"

export function AdaptiveAvgPool2DNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[160px] bg-card border-border shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-primary" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Shrink className="h-4 w-4 text-cyan-500" />
          <span className="font-medium text-sm">AdaptiveAvgPool2D</span>
        </div>
        <div className="text-xs text-muted-foreground">output: {JSON.stringify(data.output_size ?? [1, 1])}</div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-primary" />
    </Card>
  )
}
