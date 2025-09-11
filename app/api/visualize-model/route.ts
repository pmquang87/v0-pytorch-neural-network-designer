import { type NextRequest, NextResponse } from "next/server"
import { VisualizationGenerator } from "@/lib/visualization-generator"
import type { GenerateModelRequest } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const body: GenerateModelRequest = await request.json()

    if (!body.graph || !body.graph.nodes || !body.graph.edges) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid request: missing graph data",
        },
        { status: 400 },
      )
    }

    const generator = new VisualizationGenerator(body.graph)
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
