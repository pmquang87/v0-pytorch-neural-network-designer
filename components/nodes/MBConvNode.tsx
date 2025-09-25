import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Layers } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function MBConvNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, channels: 3, height: 224, width: 224 }
  const outputShape = calculateOutputShape("mbconvNode", [inputShape], data)

  return (
    <Card className="min-w-[150px] border-2 border-purple-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-purple-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-purple-500" />
          <span className="font-medium text-sm">MBConv</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>In Ch: {data.in_channels || "?"}</div>
          <div>Out Ch: {data.out_channels || "?"}</div>
          <div>Kernel: {data.kernel_size || "?"}</div>
          <div>Stride: {data.stride || "?"}</div>
          <div>Expand: {data.expand_ratio || "?"}</div>
          {data.se_ratio && <div>SE Ratio: {data.se_ratio}</div>}
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-purple-500 border-2 border-background" />
    </Card>
  )
}
