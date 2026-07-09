import { type NextRequest, NextResponse } from "next/server"
import { VisualizationGenerator } from "@/lib/visualization-generator"
import type { GenerateModelRequest } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const body: GenerateModelRequest = await request.json()

    if (!Array.isArray(body.nodes) || !Array.isArray(body.edges)) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid request: missing nodes or edges data",
        },
        { status: 400 },
      )
    }

    const graph = { nodes: body.nodes, edges: body.edges }
    const generator = new VisualizationGenerator(graph)
    const visualization = generator.generateVisualization()
    const svg = generator.generateSVG()

    return NextResponse.json({
      success: true,
      visualization,
      svg,
    })
  } catch (error) {
    console.error("Error generating visualization:", error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 },
    )
  }
}
