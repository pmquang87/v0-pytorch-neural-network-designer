import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Database } from "lucide-react"
import { formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function InputNode({ data }: { data: any }) {
  const outputShape: TensorShape = {
    batch: data.batch_size || 1,
    channels: data.channels || 3,
    height: data.height || 28,
    width: data.width || 28,
  }

  return (
    <Card className="min-w-[150px] border-2 border-primary/50 bg-card">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Database className="h-4 w-4 text-primary" />
          <span className="font-medium text-sm">Input</span>
        </div>
        <div className="text-xs text-muted-foreground">
          <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-primary border-2 border-background" />
    </Card>
  )
}
