import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { GitBranch } from "lucide-react";
import { calculateOutputShape, formatTensorShape, type TensorShape } from "@/lib/tensor-shape-calculator";
import { useEffect } from "react";

export function ConcatenateNode({ id, data }: { id: string; data: any }) {
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
  const firstValidInputShape = definedInputShapes[0];

  // Calculate output shape using the tensor shape calculator
  const outputShape = calculateOutputShape("concatenateNode", definedInputShapes, data);

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
        className="w-3 h-3 bg-indigo-500 border-2 border-background"
      />,
    );
  }

  return (
    <Card className="min-w-[160px] bg-card border-2 border-indigo-500/50 shadow-sm">
      {inputHandles}

      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="h-4 w-4 text-indigo-500" />
          <span className="font-medium text-sm">Concatenate</span>
        </div>
        <div className="text-xs text-muted-foreground">inputs: {definedInputShapes.length}</div>
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground">
            <div className="text-orange-600">
              In: {formatTensorShape(firstValidInputShape || {})}
            </div>
            <div className="text-blue-600">Out: {formatTensorShape(outputShape)}</div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-indigo-500 border-2 border-background" />
    </Card>
  );
}
