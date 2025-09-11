import { ModelGenerator } from "../lib/model-generator"
import { EXAMPLE_NETWORKS } from "../lib/example-networks"

// Test the Full U-Net model generation
const fullUNet = EXAMPLE_NETWORKS.find((network) => network.name === "Full U-Net")

if (!fullUNet) {
  console.error("Full U-Net example not found")
  process.exit(1)
}

console.log("Testing Full U-Net model generation...")
console.log(`Nodes: ${fullUNet.nodes.length}`)
console.log(`Edges: ${fullUNet.edges.length}`)

// Create the graph IR
const graphIR = {
  nodes: fullUNet.nodes,
  edges: fullUNet.edges,
}

try {
  const generator = new ModelGenerator(graphIR)
  const validation = generator.validateGraph()

  if (!validation.valid) {
    console.error("Graph validation failed:", validation.error)
    process.exit(1)
  }

  console.log("Graph validation passed")

  const generatedCode = generator.generateCode()
  console.log("\n=== GENERATED PYTORCH CODE ===")
  console.log(generatedCode)

  // Check for problematic comments
  if (generatedCode.includes("Multiple inputs - using first input for simplicity")) {
    console.log("\n⚠️  WARNING: Found 'simplicity' comments in generated code")
    const lines = generatedCode.split("\n")
    lines.forEach((line, index) => {
      if (line.includes("Multiple inputs - using first input for simplicity")) {
        console.log(`Line ${index + 1}: ${line}`)
      }
    })
  } else {
    console.log("\n✅ No problematic 'simplicity' comments found")
  }
} catch (error) {
  console.error("Model generation failed:", error)
}
