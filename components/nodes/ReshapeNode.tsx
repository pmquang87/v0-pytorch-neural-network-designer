import { Handle, Position } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { Layers } from "lucide-react";
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator";

export function ReshapeNode({ data }: { data: any }) {
  // data.inputShape is a single shape object (or undefined)
  const inputShape: TensorShape | undefined = data.inputShape;

  // The user-defined target shape for the reshape operation
  const targetShape = data.targetShape || "";

  // Calculate output shape using the tensor shape calculator
  const outputShape = calculateOutputShape("reshape", [inputShape], { targetShape });

  return (
    <Card className="min-w-[160px] bg-card border-2 border-green-500/50 shadow-sm">
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-green-500 border-2 border-background"
      />

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Layers className="h-4 w-4 text-green-500" />
          <span className="font-medium text-sm">Reshape</span>
        </div>
        <div className="text-xs text-muted-foreground">
          Target: {targetShape}
        </div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-green-600">
              In: {formatTensorShape(inputShape || {})}
            </div>
          </div>
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        className="w-3 h-3 bg-green-500 border-2 border-background"
      />
    </Card>
  );
}
