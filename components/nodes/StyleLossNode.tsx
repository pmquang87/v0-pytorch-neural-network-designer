import { Handle, Position } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { BrainCircuit } from "lucide-react";

export function StyleLossNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-2 border-purple-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-purple-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <BrainCircuit className="h-4 w-4 text-purple-500" />
          <span className="font-medium text-sm">Style Loss</span>
        </div>
        <div className="text-xs space-y-1">
          <div>Target: Gram Matrices</div>
          <div>Input: Generated Image Gram Matrices</div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-purple-500 border-2 border-background" />
    </Card>
  );
}
