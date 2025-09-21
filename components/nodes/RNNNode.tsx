import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { GitBranch } from "lucide-react"

export function RNNNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[160px] bg-card border-border shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-primary" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="h-4 w-4 text-amber-500" />
          <span className="font-medium text-sm">RNN</span>
        </div>
        <div className="text-xs text-muted-foreground">
          input: {data.input_size ?? 128}
          <br />
          hidden: {data.hidden_size ?? 64}
          <br />
          layers: {data.num_layers ?? 1}
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-primary" />
    </Card>
  )
}
