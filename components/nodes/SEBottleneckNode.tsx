import { memo } from "react"
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

function SEBottleneckNodeImpl({ data }: NodeProps) {
  return (
    <Card className="w-56">
      <CardHeader className="p-2">
        <CardTitle className="text-sm text-center">SE Bottleneck</CardTitle>
      </CardHeader>
      <CardContent className="p-2 text-xs text-center">
        <p>In-Planes: {String(data.in_planes)}</p>
        <p>Planes: {String(data.planes)}</p>
        <p>Stride: {String(data.stride)}</p>
        <p>Downsample: {data.downsample ? 'Yes' : 'No'}</p>
      </CardContent>
      <Handle type="target" position={Position.Left} />
      <Handle type="source" position={Position.Right} />
    </Card>
  );
}

export const SEBottleneckNode = memo(SEBottleneckNodeImpl)
SEBottleneckNode.displayName = "SEBottleneckNode"
