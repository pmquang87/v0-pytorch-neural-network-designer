import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Layers } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function DepthwiseConv2DNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { channels: 32, height: 28, width: 28 }
  const outputShape = calculateOutputShape("depthwiseconv2dNode", [inputShape], data)

  return (
    <Card className="min-w-[180px] border-2 border-orange-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-orange-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-orange-500" />
          <span className="font-medium text-sm">DepthwiseConv2D</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>In Channels: {data.in_channels || "?"}</div>
          <div>Out Channels: {data.out_channels || "?"}</div>
          <div>Kernel Size: {data.kernel_size || "?"}</div>
          <div>Groups: {data.groups || "?"}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-orange-500 border-2 border-background" />
    </Card>
  )
}
