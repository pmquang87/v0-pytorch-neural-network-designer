import { Handle, Position } from "@xyflow/react"
import { Card } from "@/components/ui/card"
import { Layers } from "lucide-react"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

export function TransformerEncoderLayerNode({ data }: { data: any }) {
  return (
    <Card className="w-64 bg-card border-purple-500/50 shadow-sm">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-primary" />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-purple-500" />
          <span className="font-medium text-sm">Transformer Encoder Layer</span>
        </div>
        <div className="text-xs text-muted-foreground">
          d_model: {data.d_model ?? 512}
          <br />
          nhead: {data.nhead ?? 8}
        </div>
        {data.inputShape && (
          <div className="text-xs text-gray-500 mt-1">In: {formatTensorShape(data.inputShape)}</div>
        )}
        {data.outputShape && (
          <div className="text-xs text-gray-500">Out: {formatTensorShape(data.outputShape)}</div>
        )}
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-primary" />
    </Card>
  )
}
