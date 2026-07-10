import { memo } from "react"
import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { AppWindow } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

function ConstantNodeImpl({ data }: { data: any }) {
  const outputShape = calculateOutputShape("constantNode", [], data)

  return (
    <Card className="min-w-[150px] border-2 border-green-500/50 bg-card">
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <AppWindow className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Constant</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Channels: {data.channels || "?"}</div>
          <div>Height: {data.height || "?"}</div>
          <div>Width: {data.width || "?"}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-green-500 border-2 border-background" />
    </Card>
  )
}

export const ConstantNode = memo(ConstantNodeImpl)
ConstantNode.displayName = "ConstantNode"
