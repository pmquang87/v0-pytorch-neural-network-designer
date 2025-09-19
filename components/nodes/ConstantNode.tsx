import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Box } from "lucide-react"
import { formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function ConstantNode({ data }: { data: any }) {
  const outputShape: TensorShape = {
    batch: 1,
    channels: data.channels || 1,
    height: data.height || 1,
    width: data.width || 1,
  }

  return (
    <Card className="min-w-[150px] border-2 border-green-500/50 bg-card">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Box className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Constant</span>
        </div>
        <div className="text-xs text-muted-foreground">
          <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-green-500 border-2 border-background" />
    </Card>
  )
}
