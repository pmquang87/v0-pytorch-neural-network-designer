import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { LogIn } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function InputNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[160px] bg-card border-green-500/50 shadow-sm">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <LogIn className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Input</span>
        </div>
        <div className="text-xs text-muted-foreground">
          Shape: {formatTensorShape(data)}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-green-500 border-2 border-background"
      />
    </Card>
  )
}
