import { memo } from "react"
import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Boxes } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

function MoENodeImpl({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape
  const outputShape = calculateOutputShape("moeNode", [inputShape], data)

  return (
    <Card className="min-w-[170px] border-2 border-fuchsia-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-fuchsia-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Boxes className="h-4 w-4 text-fuchsia-500" />
          <span className="font-medium text-sm">MoE</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>d_model: {data.d_model ?? "?"}</div>
          <div>experts: {data.num_experts ?? "?"} (top-{data.top_k ?? 2})</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs space-y-1">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-fuchsia-500 border-2 border-background" />
    </Card>
  )
}

export const MoENode = memo(MoENodeImpl)
MoENode.displayName = "MoENode"
