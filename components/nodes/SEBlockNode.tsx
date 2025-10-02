import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export function SEBlockNode({ data }: NodeProps) {
  return (
    <Card className="w-48">
      <CardHeader className="p-2">
        <CardTitle className="text-sm text-center">SE Block</CardTitle>
      </CardHeader>
      <CardContent className="p-2 text-xs text-center">
        <p>In: {data.in_channels}</p>
        <p>Reduction: {data.reduction}</p>
      </CardContent>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </Card>
  );
}
