import { Handle, Position } from "@xyflow/react";
import { Card } from "@/components/ui/card";
import { GitBranch } from "lucide-react";

export function ChunkNode({ data }: { data: any }) {
  return (
    <Card className="min-w-[150px] border-2 border-indigo-500/50 bg-card">
      <Handle type="target" position={Position.Left} className="w-3 h-3 bg-indigo-500 border-2 border-background" />
      <div className="p-3">
        <div className="flex items-center gap-2 mb-2">
          <GitBranch className="h-4 w-4 text-indigo-500" />
          <span className="font-medium text-sm">Chunk</span>
        </div>
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Chunks: {data.chunks || 2}</div>
          <div>Dim: {data.dim || -1}</div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        id="output1"
        className="w-3 h-3 bg-indigo-500 border-2 border-background"
        style={{ top: "33%" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="output2"
        className="w-3 h-3 bg-indigo-500 border-2 border-background"
        style={{ top: "66%" }}
      />
    </Card>
  );
}
