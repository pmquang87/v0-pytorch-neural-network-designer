import { Handle, Position } from "@xyflow/react"
import type { NodeData, HelpContent } from "@/lib/types"
import { HelpCircle } from "lucide-react"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { formatTensorShape } from "@/lib/tensor-shape-calculator"

interface SelectNodeData extends NodeData {
  dim?: number
  index?: number
}

interface SelectNodeProps {
  data: SelectNodeData
}

const selectNodeHelp: HelpContent = {
  title: "Select Node",
  description: "Selects a specific dimension or index from the input tensor. Useful for tensor slicing operations.",
  parameters: [
    {
      name: "dim",
      type: "number",
      description: "Dimension along which to select. For example, dim=0 selects along the batch dimension.",
      default: 0
    },
    {
      name: "index",
      type: "number",
      description: "Index to select within the specified dimension.",
      default: 0
    }
  ],
  examples: [
    "Select the first channel of an image tensor: dim=1, index=0",
    "Select a specific sample from a batch: dim=0, index=5"
  ],
  tips: [
    "Ensure the index is within the bounds of the dimension size.",
    "The output shape will be the same as input shape with the selected dimension reduced to 1."
  ]
}

export function SelectNode({ data }: SelectNodeProps) {
  return (
    <div className="bg-purple-100 border-2 border-purple-300 rounded-lg p-3 min-w-[120px]">
      <Handle type="target" position={Position.Left} />

      <div className="text-center relative">
        <div className="font-semibold text-purple-800 text-sm mb-1">Select</div>
        <div className="absolute top-0 right-0">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button className="text-purple-600 hover:text-purple-800 transition-colors">
                  <HelpCircle size={16} />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <p className="font-bold">{selectNodeHelp.title}</p>
                <p>{selectNodeHelp.description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className="text-xs text-purple-600">
          dim: {data.dim ?? 0}, idx: {data.index ?? 0}
        </div>
        {data.inputShape && <div className="text-xs text-gray-500 mt-1">In: {formatTensorShape(data.inputShape)}</div>}
        {data.outputShape && <div className="text-xs text-gray-500">Out: {formatTensorShape(data.outputShape)}</div>}
      </div>

      <Handle type="source" position={Position.Right} />
    </div>
  )
}
