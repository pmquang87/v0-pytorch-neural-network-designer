import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { GitBranch } from "lucide-react"
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator"

export function AdaptiveInstanceNormNode({ data }: { data: any }) {
  const inputShape: TensorShape = data.inputShape || { batch: 1, features: 512, height: 4, width: 4 }
  const styleShape: TensorShape = data.styleShape || { batch: 1, features: 512 }
  const outputShape = calculateOutputShape("adaptiveInstanceNormNode", [inputShape, styleShape], data)

  return (
    <Card className="min-w-[200px] border-2 border-yellow-500/50 bg-card">
      <Handle
        type="target"
        id="input"
        position={Position.Left}
        style={{ top: "30%" }}
        className="w-3 h-3 bg-yellow-500 border-2 border-background"
      />
      <Handle
        type="target"
        id="style"
        position={Position.Left}
        style={{ top: "70%" }}
        className="w-3 h-3 bg-yellow-500 border-2 border-background"
      />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="h-4 w-4 text-yellow-500" />
          <span className="font-medium text-sm">Adaptive Instance Norm</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Num Features: {data.num_features || "?"}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">In: {formatTensorShape(inputShape)}</div>
            <div className="text-purple-600">Style: {formatTensorShape(styleShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-yellow-500 border-2 border-background" />
    </Card>
  )
}
