import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Network } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function LayerNormNode({ data }: { data: any }) {
  return (
    <Card className="w-56 border-2 border-gray-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-gray-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Network className="h-4 w-4 text-gray-500" />
          <span className="font-medium text-sm">LayerNorm</span>
        </div>
        <div className="text-xs text-muted-foreground">
          <div>Shape: {data.normalized_shape}</div>
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs space-y-1">
            <div className="text-green-600">In: {formatTensorShape(data.inputShape)}</div>
            <div className="text-blue-600">Out: {formatTensorShape(data.outputShape)}</div>
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-gray-500 border-2 border-background" />
    </Card>
  )
}
