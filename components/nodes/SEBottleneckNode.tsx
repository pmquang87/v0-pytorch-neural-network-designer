import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export function SEBottleneckNode({ data }: NodeProps) {
  return (
    <Card className="w-56">
      <CardHeader className="p-2">
        <CardTitle className="text-sm text-center">SE Bottleneck</CardTitle>
      </CardHeader>
      <CardContent className="p-2 text-xs text-center">
        <p>In-Planes: {data.in_planes}</p>
        <p>Planes: {data.planes}</p>
        <p>Stride: {data.stride}</p>
        <p>Downsample: {data.downsample ? 'Yes' : 'No'}</p>
      </CardContent>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </Card>
  );
}
