import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { BarChart3 } from "lucide-react"

export function GroupNormNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[160px] bg-card border-border shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-primary" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="h-4 w-4 text-violet-500" />
          <span className="font-medium text-sm">GroupNorm</span>
        </div>
        <div className="text-xs text-muted-foreground">
          groups: {data.num_groups ?? 32}
          <br />
          channels: {data.num_channels ?? 128}
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-primary" />
    </Card>
  )
}
