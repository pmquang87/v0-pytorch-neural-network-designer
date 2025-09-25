import { Handle, Position } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { Box } from "lucide-react";

export function DefaultNode({ data }: { data: any }) {
  return (
    <Card className="w-48 border-2 border-gray-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-gray-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <Box className="h-4 w-4 text-gray-500" />
          <span className="font-medium text-sm">{data.label || 'Custom Node'}</span>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="w-3 h-3 bg-gray-500 border-2 border-background" />
    </Card>
  );
}
