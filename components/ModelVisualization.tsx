import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface ModelVisualizationProps {
  svg: string
  className?: string
}

export function ModelVisualization({ svg, className }: ModelVisualizationProps) {
  return (
    <Card className={`p-4 ${className}`}>
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">Model Architecture</h3>
        <Badge variant="secondary" className="text-xs">
          Generated Visualization
        </Badge>
      </div>
      <ScrollArea className="h-[400px] w-full">
        <div className="flex justify-center" dangerouslySetInnerHTML={{ __html: svg }} />
      </ScrollArea>
    </Card>
  )
}
