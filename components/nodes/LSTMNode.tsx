import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { GitBranch } from "lucide-react"

export function LSTMNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[150px] border-2 border-orange-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-orange-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">LSTM</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Input: {data.input_size || "?"}</div>
          <div>Hidden: {data.hidden_size || "?"}</div>
          <div>Layers: {data.num_layers || 1}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500 border-2 border-background" />
    </Card>
  )
}
