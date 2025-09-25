import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { X } from "lucide-react";
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator";
import { useEffect } from "react";

export function MultiplyNode({ id, data }: { id: string; data: any }) {
  const updateNodeInternals = useUpdateNodeInternals();
  // data.inputShape is an array of shapes (or undefined) populated by propagateTensorShapes
  const inputShapes: (TensorShape | undefined)[] = Array.isArray(data.inputShape) ? data.inputShape : [];

  // The number of handles to render is determined by the length of the inputShapes array,
  // which is dynamically calculated in the main page component. Default to 2 for a new node.
  const numInputs = inputShapes.length > 0 ? inputShapes.length : 2;

  useEffect(() => {
    updateNodeInternals(id);
  }, [id, numInputs, updateNodeInternals]);

  // Get all defined input shapes for output calculation and display
  const definedInputShapes = inputShapes.filter((s): s is TensorShape => s && Object.keys(s).length > 0);

  // Calculate output shape using the tensor shape calculator
  const outputShape = calculateOutputShape("multiplyNode", definedInputShapes, data);

  const inputHandles = [];
  for (let i = 1; i <= numInputs; i++) {
    const topPercentage = (i / (numInputs + 1)) * 100;
    inputHandles.push(
      <Handle
        key={`input${i}`}
        type="target"
        position={Position.Left}
        id={`input${i}`}
        style={{ top: `${topPercentage}%` }}
        className="w-3 h-3 bg-purple-500 border-2 border-background"
      />,
    );
  }

  return (
    <Card className="min-w-[160px] bg-card border-2 border-purple-500/50 shadow-sm">
      {inputHandles}

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <X className="h-4 w-4 text-purple-500" />
          <span className="font-medium text-sm">Multiply</span>
        </div>
        <div className="text-xs text-muted-foreground">inputs: {definedInputShapes.length}</div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            {definedInputShapes.map((shape, i) => (
              <div key={i} className="text-orange-600">
                In {i + 1}: {formatTensorShape(shape)}
              </div>
            ))}
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-purple-500 border-2 border-background" />
    </Card>
  );
}
