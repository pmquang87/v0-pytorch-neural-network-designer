import { type NextRequest, NextResponse } from "next/server"
import { ModelGenerator } from "@/lib/model-generator"
import type { GenerateModelRequest, GenerateModelResponse } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const body: GenerateModelRequest = await request.json()

    if (!body.nodes || !body.edges) {
      return NextResponse.json(
        {
          success: false,
          error: "Invalid request: missing nodes or edges data",
        } as GenerateModelResponse,
        { status: 400 },
      )
    }

    const graph = { nodes: body.nodes, edges: body.edges }
    const generator = new ModelGenerator(graph)
    const code = generator.generateCode()

    return NextResponse.json({
      success: true,
      code,
    } as GenerateModelResponse)
  } catch (error) {
    console.error("Error generating model:", error)
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error occurred",
      } as GenerateModelResponse,
      { status: 500 },
    )
  }
}
