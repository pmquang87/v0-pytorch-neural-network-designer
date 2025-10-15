import { type NextRequest, NextResponse } from "next/server"
import { ModelGenerator } from "@/lib/model-generator"
import type { GenerateModelRequest } from "@/lib/types"

export async function POST(request: NextRequest) {
  try {
    const body: GenerateModelRequest = await request.json()

    if (!body.nodes || !body.edges) {
      return NextResponse.json(
        {
          valid: false,
          error: "Invalid request: missing nodes or edges data",
        },
        { status: 400 },
      )
    }

    const graph = { nodes: body.nodes, edges: body.edges }
    const generator = new ModelGenerator(graph)
    const validation = generator.validateGraph()

    return NextResponse.json(validation)
  } catch (error) {
    console.error("Error validating graph:", error)
    return NextResponse.json(
      {
        valid: false,
        error: error instanceof Error ? error.message : "Unknown error occurred",
      },
      { status: 500 },
    )
  }
}
