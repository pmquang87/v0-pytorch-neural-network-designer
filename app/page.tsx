"use client"

import type React from "react"

import { useState, useCallback, useEffect, useRef } from "react"
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Controls,
  Background,
  addEdge,
  applyNodeChanges,
  applyEdgeChanges,
  type Connection,
  type Edge,
  type Node,
  type NodeTypes,
  BackgroundVariant,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"

import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useToast } from "@/hooks/use-toast"
import {
  Brain,
  Code,
  RotateCcw,
  Undo2,
  Redo2,
  Network,
  HelpCircle,
  BarChart3,
  Loader2,
  Download,
  Copy,
  Zap,
  Shrink,
  Eye,
  Plus,
  GitBranch,
  Database,
  Layers,
  AlertTriangle,
  Save,
  FolderOpen,
  Trash2,
  CheckCircle,
  Box,
  FileUp,
} from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { ResizableBox } from "react-resizable"
import "react-resizable/css/styles.css"

import { EXAMPLE_NETWORKS } from "@/lib/example-networks"
import {
  calculateOutputShape as calcOutputShape,
  formatTensorShape,
  type TensorShape,
} from "@/lib/tensor-shape-calculator"

import { analyzeModel, formatNumber, type ModelAnalysis } from "@/lib/model-analyzer"
import { useAutoSave, StorageUtils } from "@/lib/auto-save"
import { useModelValidation } from "@/lib/model-validator"
import { useUndoRedo } from "@/lib/undo-redo"
import { Input } from "@/components/ui/input"
import { EditableNumberInput } from "@/components/ui/EditableNumberInput"

// Custom Node Components
import { InputNode } from "@/components/nodes/InputNode"
import { ConstantNode } from "@/components/nodes/ConstantNode"
import { LinearNode } from "@/components/nodes/LinearNode"
import { Conv2DNode } from "@/components/nodes/Conv2DNode"
import { ReLUNode } from "@/components/nodes/ReLUNode"
import { SigmoidNode } from "@/components/nodes/SigmoidNode"
import { TanhNode } from "@/components/nodes/TanhNode"
import { SoftmaxNode } from "@/components/nodes/SoftmaxNode"
import { LeakyReLUNode } from "@/components/nodes/LeakyReLUNode"
import { DropoutNode } from "@/components/nodes/DropoutNode"
import { FlattenNode } from "@/components/nodes/FlattenNode"
import { MaxPool2DNode } from "@/components/nodes/maxpool2dNode"
import { AvgPool2DNode } from "@/components/nodes/AvgPool2DNode"
import { AdaptiveAvgPool2DNode } from "@/components/nodes/AdaptiveAvgPool2DNode"
import { BatchNorm1DNode } from "@/components/nodes/BatchNorm1DNode"
import { BatchNorm2DNode } from "@/components/nodes/BatchNorm2DNode"
import { LayerNormNode } from "@/components/nodes/LayerNormNode"
import { GroupNormNode } from "@/components/nodes/GroupNormNode"
import { ConcatenateNode } from "@/components/nodes/ConcatenateNode"
import { AddNode } from "@/components/nodes/AddNode"
import { LSTMNode } from "@/components/nodes/LSTMNode"
import { GRUNode } from "@/components/nodes/GRUNode"
import { Conv1DNode } from "@/components/nodes/Conv1DNode"
import { Conv3DNode } from "@/components/nodes/Conv3DNode"
import { ConvTranspose1DNode } from "@/components/nodes/ConvTranspose1DNode"
import { ConvTranspose2DNode } from "@/components/nodes/ConvTranspose2DNode"
import { ConvTranspose3DNode } from "@/components/nodes/ConvTranspose3DNode"
import { DepthwiseConv2DNode } from "@/components/nodes/DepthwiseConv2DNode"
import { SeparableConv2DNode } from "@/components/nodes/SeparableConv2DNode"
import { GELUNode } from "@/components/nodes/GELUNode"
import { SiLUNode } from "@/components/nodes/SiLUNode"
import { MishNode } from "@/components/nodes/MishNode"
import { HardswishNode } from "@/components/nodes/HardswishNode"
import { HardsigmoidNode } from "@/components/nodes/HardsigmoidNode"
import { InstanceNorm1DNode } from "@/components/nodes/InstanceNorm1DNode"
import { InstanceNorm2DNode } from "@/components/nodes/InstanceNorm2DNode"
import { InstanceNorm3DNode } from "@/components/nodes/InstanceNorm3DNode"
import { MultiheadAttentionNode } from "@/components/nodes/MultiheadAttentionNode"
import { TransformerEncoderLayerNode } from "@/components/nodes/TransformerEncoderLayerNode"
import { TransformerDecoderLayerNode } from "@/components/nodes/TransformerDecoderLayerNode"
import { TransposeNode } from "@/components/nodes/TransposeNode"
import { SelectNode } from "@/components/nodes/SelectNode"

const initialNodes: Node[] = [
  {
    id: "input-1",
    type: "inputNode",
    position: { x: 100, y: 100 },
    data: { channels: 3, height: 28, width: 28 },
  },
]

const initialEdges: Edge[] = []

const nodeTypes: NodeTypes = {
  inputNode: InputNode,
  constantNode: ConstantNode,
  linearNode: LinearNode,
  conv2dNode: Conv2DNode,
  conv1dNode: Conv1DNode,
  conv3dNode: Conv3DNode,
  convtranspose1dNode: ConvTranspose1DNode,
  convtranspose2dNode: ConvTranspose2DNode,
  convtranspose3dNode: ConvTranspose3DNode,
  depthwiseconv2dNode: DepthwiseConv2DNode,
  separableconv2dNode: SeparableConv2DNode,
  reluNode: ReLUNode,
  sigmoidNode: SigmoidNode,
  tanhNode: TanhNode,
  softmaxNode: SoftmaxNode,
  leakyreluNode: LeakyReLUNode,
  geluNode: GELUNode,
  siluNode: SiLUNode,
  mishNode: MishNode,
  hardswishNode: HardswishNode,
  hardsigmoidNode: HardsigmoidNode,
  dropoutNode: DropoutNode,
  flattenNode: FlattenNode,
  maxpool2dNode: MaxPool2DNode,
  avgpool2dNode: AvgPool2DNode,
  adaptiveavgpool2dNode: AdaptiveAvgPool2DNode,
  batchnorm1dNode: BatchNorm1DNode,
  batchnorm2dNode: BatchNorm2DNode,
  layernormNode: LayerNormNode,
  groupnormNode: GroupNormNode,
  instancenorm1dNode: InstanceNorm1DNode,
  instancenorm2dNode: InstanceNorm2DNode,
  instancenorm3dNode: InstanceNorm3DNode,
  concatenateNode: ConcatenateNode,
  addNode: AddNode,
  lstmNode: LSTMNode,
  gruNode: GRUNode,
  multiheadattentionNode: MultiheadAttentionNode,
  transformerencoderlayerNode: TransformerEncoderLayerNode,
  transformerdecoderlayerNode: TransformerDecoderLayerNode,
  transposeNode: TransposeNode,
  selectNode: SelectNode,
}

export default function NeuralNetworkDesigner() {
  const propertiesPanelRef = useRef<HTMLDivElement>(null)
  const reactFlowWrapper = useRef<HTMLDivElement>(null)
  const {
    nodes,
    setNodes,
    edges,
    setEdges,
    undo,
    redo,
    takeSnapshot,
    canUndo,
    canRedo,
  } = useUndoRedo(initialNodes, initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const onNodesChange = useCallback(
    (changes) => {
      const removals = changes.filter((c) => c.type === "remove")
      if (removals.length > 0) {
        takeSnapshot()
        if (selectedNode && removals.some((r) => r.id === selectedNode.id)) {
          setSelectedNode(null)
        }
      }
      setNodes((nds) => applyNodeChanges(changes, nds))
    },
    [setNodes, takeSnapshot, selectedNode, setSelectedNode],
  )
  const onEdgesChange = useCallback(
    (changes) => {
      if (changes.some((c) => c.type === "remove")) {
        takeSnapshot()
      }
      setEdges((eds) => applyEdgeChanges(changes, eds))
    },
    [setEdges, takeSnapshot],
  )
  const [showCodeDialog, setShowCodeDialog] = useState(false)
  const [generatedCode, setGeneratedCode] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  const { toast } = useToast()
  const isUpdatingShapes = useRef(false)
  const [copySuccess, setCopySuccess] = useState(false)

  const [modelAnalysis, setModelAnalysis] = useState<ModelAnalysis | null>(null)
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false)

  const [showHelpDialog, setShowHelpDialog] = useState(false)

  const [showCodeInputDialog, setShowCodeInputDialog] = useState(false)
  const [inputCode, setInputCode] = useState("")
  const [parseErrors, setParseErrors] = useState<string[]>([])
  const [parseWarnings, setParseWarnings] = useState<string[]>([])
  const [unsupportedModules, setUnsupportedModules] = useState<string[]>([])

  const [validationResults, setValidationResults] = useState<{ errors: string[]; warnings: string[] } | null>(null)
  const [liveValidationResults, setLiveValidationResults] = useState<{ errors: string[]; warnings: string[] } | null>(
    null,
  )
  const [showValidationPanel, setShowValidationPanel] = useState(false)

  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [showLoadDialog, setShowLoadDialog] = useState(false)
  const [modelName, setModelName] = useState("")
  const [currentModelName, setCurrentModelName] = useState<string | null>(null)
  const [savedModels, setSavedModels] = useState<Array<{ key: string; name: string; timestamp: number }>>([])

  const reactFlowInstanceRef = useRef<any>(null)
  const autoSave = useAutoSave()
  const { validateModel } = useModelValidation()

  const isInputConnected = selectedNode ? edges.some((edge) => edge.target === selectedNode.id) : false

  useEffect(() => {
    const handleAutoSaveRequest = () => {
      if (typeof window !== "undefined") {
        const event = new CustomEvent("auto-save-trigger", { detail: { nodes, edges } })
        window.dispatchEvent(event)
      }
    }

    window.addEventListener("auto-save-request", handleAutoSaveRequest)

    return () => {
      window.removeEventListener("auto-save-request", handleAutoSaveRequest)
    }
  }, [nodes, edges])

  useEffect(() => {
    if (autoSave.hasSavedData()) {
      const loadedState = autoSave.load()
      if (loadedState) {
        setNodes(loadedState.nodes)
        setEdges(loadedState.edges)
        toast({
          title: "Session Restored",
          description: "Your previous session has been loaded.",
        })
      }
    }
  }, [])

  useEffect(() => {
    const results = validateModel(nodes, edges)
    setLiveValidationResults(results)
  }, [nodes, edges, validateModel])

  const propagateTensorShapes = useCallback(() => {
    if (isUpdatingShapes.current) return
    isUpdatingShapes.current = true

    setNodes((currentNodes) => {
      const updatedNodes = currentNodes.map((node) => ({ ...node, data: { ...node.data } }))
      const nodeMap = new Map(updatedNodes.map((node) => [node.id, node]))

      const visited = new Set<string>()
      const processing = new Set<string>()
      const sorted: string[] = []

      const visit = (nodeId: string) => {
        if (processing.has(nodeId)) return
        if (visited.has(nodeId)) return
        processing.add(nodeId)
        const outgoingEdges = edges.filter((edge) => edge.source === nodeId)
        for (const edge of outgoingEdges) {
          visit(edge.target)
        }
        processing.delete(nodeId)
        visited.add(nodeId)
        sorted.unshift(nodeId)
      }

      for (const node of updatedNodes) {
        visit(node.id)
      }

      for (const nodeId of sorted) {
        const node = nodeMap.get(nodeId)
        if (!node) continue

        if (node.type === "inputNode" || node.type === "constantNode") {
          const isInput = node.type === "inputNode"
          const shape: TensorShape = {
            channels: (node.data as any).channels ?? (isInput ? 3 : 1),
            height: (node.data as any).height ?? (isInput ? 28 : 1),
            width: (node.data as any).width ?? (isInput ? 28 : 1),
          }
          node.data = { ...node.data, inputShape: shape, outputShape: shape }
          continue
        }

        const inputEdges = edges.filter((edge) => edge.target === nodeId)
        let allInputShapes: (TensorShape | undefined)[] = []

        if (node.type === "concatenateNode" || node.type === "addNode") {
          const numInputs = node.data.num_inputs || 2
          for (let i = 1; i <= numInputs; i++) {
            const handleId = `input${i}`
            const edge = inputEdges.find((e) => e.targetHandle === handleId)
            if (edge) {
              const sourceNode = nodeMap.get(edge.source)
              allInputShapes.push(sourceNode?.data.outputShape)
            } else {
              allInputShapes.push(undefined)
            }
          }
        } else {
          allInputShapes = inputEdges.map((edge) => {
            const sourceNode = nodeMap.get(edge.source)
            return sourceNode?.data.outputShape
          })
        }

        const cleanInputShapes = allInputShapes.filter((s): s is TensorShape => !!s)

        const data = { ...node.data }
        const firstInputShape = cleanInputShapes[0]

        if (firstInputShape && firstInputShape.channels) {
          if (
            node.type === "conv1dNode" ||
            node.type === "conv2dNode" ||
            node.type === "conv3dNode" ||
            node.type === "convtranspose1dNode" ||
            node.type === "convtranspose2dNode" ||
            node.type === "convtranspose3dNode" ||
            node.type === "depthwiseconv2dNode" ||
            node.type === "separableconv2dNode"
          ) {
            data.in_channels = firstInputShape.channels
          } else if (
            node.type === "batchnorm1dNode" ||
            node.type === "batchnorm2dNode" ||
            node.type === "instancenorm1dNode" ||
            node.type === "instancenorm2dNode" ||
            node.type === "instancenorm3dNode"
          ) {
            data.num_features = firstInputShape.channels
          } else if (node.type === "groupnormNode") {
            data.num_channels = firstInputShape.channels
          }
        }

        if (node.type === "linearNode" && firstInputShape) {
          if (typeof firstInputShape.features === "number") {
            data.in_features = firstInputShape.features
          } else if (firstInputShape.features === undefined) {
            const flattenedSize =
              (firstInputShape.channels || 1) * (firstInputShape.height || 1) * (firstInputShape.width || 1)
            data.in_features = flattenedSize
          }
        }

        const outputShape = calcOutputShape(node.type || "", cleanInputShapes, data)

        if (node.type === "concatenateNode" || node.type === "addNode") {
          node.data = {
            ...data,
            inputShape: allInputShapes,
            outputShape,
          }
        } else {
          node.data = {
            ...data,
            inputShape: cleanInputShapes[0],
            outputShape,
          }
        }
      }

      isUpdatingShapes.current = false
      return updatedNodes
    })
  }, [edges, setNodes])

  const onConnect = useCallback(
    (params: Connection) => {
      takeSnapshot()
      setEdges((eds) => addEdge(params, eds))
      toast({
        title: "Connection Created",
        description: "Successfully connected layers",
      })
    },
    [setEdges, toast, takeSnapshot],
  )

  const addNode = useCallback(
    (type: string, data: any = {}) => {
      takeSnapshot()
      const reactFlowInstance = reactFlowInstanceRef.current;
      let position = { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 };

      if (reactFlowInstance && reactFlowWrapper.current) {
        const rect = reactFlowWrapper.current.getBoundingClientRect();
        const targetPosition = reactFlowInstance.screenToFlowPosition({
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        });

        // Offset by half of an average node size to center the node itself
        targetPosition.x -= 75;
        targetPosition.y -= 25;

        position = targetPosition;
      }

      const newNode: Node = {
        id: `${type}_${Date.now()}`,
        type,
        position,
        data,
      }
      setNodes((nds) => [...nds, newNode])
    },
    [setNodes, takeSnapshot],
  )

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setNodes((currentNodes) => {
        const currentNode = currentNodes.find((n) => n.id === node.id)
        if (currentNode) {
          setSelectedNode(currentNode)
        }
        return currentNodes
      })
    },
    [setNodes],
  )

  const updateNodeData = useCallback(
    (nodeId: string, newData: any) => {
      takeSnapshot()
      setNodes((nodes) => {
        const updatedNodes = nodes.map((node) => {
          if (node.id === nodeId) {
            const updatedNode = {
              ...node,
              data: { ...node.data, ...newData },
            }
            if (selectedNode && selectedNode.id === nodeId) {
              setSelectedNode(updatedNode)
            }
            return updatedNode
          }
          return node
        })
        return updatedNodes
      })
    },
    [setNodes, selectedNode, takeSnapshot],
  )

  useEffect(() => {
    const timer = setTimeout(() => propagateTensorShapes(), 50)
    return () => clearTimeout(timer)
  }, [nodes, edges, propagateTensorShapes])

  const loadExample = useCallback(
    (example: any) => {
      try {
        const invalidNodeTypes = example.nodes.filter((node: any) => !nodeTypes[node.type])
        if (invalidNodeTypes.length > 0) {
          toast({
            title: "Loading Failed",
            description: `Example contains unsupported node types: ${invalidNodeTypes.map((n: any) => n.type).join(", ")}`,
            variant: "destructive",
          })
          return
        }

        const nodeIds = new Set(example.nodes.map((node: any) => node.id))
        const invalidEdges = example.edges.filter((edge: any) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target))
        if (invalidEdges.length > 0) {
          toast({
            title: "Loading Failed",
            description: "Example contains invalid connections",
            variant: "destructive",
          })
          return
        }

        takeSnapshot()
        setNodes(example.nodes)
        setEdges(example.edges)
        setSelectedNode(null)
        setCurrentModelName(example.name)

        toast({
          title: "Example Loaded",
          description: `Successfully loaded ${example.name}`,
        })

        setTimeout(() => {
          if (reactFlowInstanceRef.current) {
            reactFlowInstanceRef.current.fitView({ padding: 0.1, duration: 800 })
          }
        }, 100)
      } catch (error) {
        toast({
          title: "Loading Failed",
          description: `Failed to load ${example.name}: ${error}`,
          variant: "destructive",
        })
      }
    },
    [setNodes, setEdges, toast, nodeTypes, setCurrentModelName, takeSnapshot],
  )

  const resetCanvas = useCallback(() => {
    takeSnapshot()
    setNodes(initialNodes)
    setEdges(initialEdges)
    setSelectedNode(null)
    setCurrentModelName(null)
    toast({
      title: "Canvas Reset",
      description: "Canvas has been reset to initial state",
    })
  }, [setNodes, setEdges, toast, setCurrentModelName, takeSnapshot])

  const analyzeCurrentModel = useCallback(() => {
    if (nodes.length === 0) return
    const analysis = analyzeModel(nodes, edges)
    setModelAnalysis(analysis)
    setShowAnalysisPanel(true)
  }, [nodes, edges])

  const handleValidateModel = useCallback(() => {
    const results = validateModel(nodes, edges)
    setValidationResults(results)
    setShowValidationPanel(true)
  }, [nodes, edges, validateModel])

  const handleSaveModel = useCallback(() => {
    if (!modelName) {
      toast({ title: "Error", description: "Model name cannot be empty.", variant: "destructive" })
      return
    }
    StorageUtils.saveModel(modelName, nodes, edges)
    setCurrentModelName(modelName)
    toast({ title: "Model Saved", description: `Model \"${modelName}\" has been saved.` })
    setShowSaveDialog(false)
    setModelName("")
  }, [modelName, nodes, edges, toast, setCurrentModelName])

  const handleExportModel = useCallback(() => {
    if (!modelName) {
      toast({ title: "Error", description: "Please enter a model name to export.", variant: "destructive" });
      return;
    }
    const modelData = {
      name: modelName,
      nodes,
      edges,
    };
    const jsonString = JSON.stringify(modelData, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${modelName.replace(/\s+/g, '_')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast({ title: "Model Exported", description: `Model "${modelName}" has been exported as a JSON file.` });
    setShowSaveDialog(false);
  }, [modelName, nodes, edges, toast, setShowSaveDialog]);

  const handleImportModel = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result;
        if (typeof text !== 'string') {
          throw new Error("File is not a valid text file.");
        }
        const importedData = JSON.parse(text);

        if (importedData && Array.isArray(importedData.nodes) && Array.isArray(importedData.edges)) {
          takeSnapshot();
          setNodes(importedData.nodes);
          setEdges(importedData.edges);
          const importedModelName = importedData.name || file.name.replace('.json', '');
          setCurrentModelName(importedModelName);
          toast({ title: "Model Imported", description: `Successfully imported "${importedModelName}".` });
          setShowLoadDialog(false);
        } else {
          throw new Error("Invalid model file format.");
        }
      } catch (error) {
        toast({
          title: "Import Failed",
          description: error instanceof Error ? error.message : "Could not parse the model file.",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
    event.target.value = ''; // Reset file input
  }, [setNodes, setEdges, toast, takeSnapshot, setCurrentModelName, setShowLoadDialog]);

  const handleOpenLoadDialog = useCallback(() => {
    setSavedModels(StorageUtils.getAllModels())
    setShowLoadDialog(true)
  }, [])

  const handleLoadModel = useCallback(
    (key: string) => {
      const loadedState = StorageUtils.loadModel(key)
      if (loadedState) {
        const model = savedModels.find((m) => m.key === key)
        if (model) {
          setCurrentModelName(model.name)
        }
        takeSnapshot()
        setNodes(loadedState.nodes)
        setEdges(loadedState.edges)
        toast({ title: "Model Loaded", description: `Model has been loaded successfully.` })
        setShowLoadDialog(false)

        setTimeout(() => {
          if (reactFlowInstanceRef.current) {
            reactFlowInstanceRef.current.fitView({ padding: 0.1, duration: 800 })
          }
        }, 100)
      }
    },
    [setNodes, setEdges, toast, savedModels, setCurrentModelName, takeSnapshot],
  )

  const handleDeleteModel = useCallback((key: string) => {
    StorageUtils.deleteModel(key)
    setSavedModels(StorageUtils.getAllModels())
    toast({ title: "Model Deleted", description: "The model has been deleted." })
  }, [])

  const generateModel = useCallback(async () => {
    setIsGenerating(true)
    try {
      const response = await fetch("/api/generate-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodes, edges }),
      })

      if (!response.ok) {
        throw new Error("Failed to generate model")
      }

      const data = await response.json()
      setGeneratedCode(data.code)

      setShowCodeDialog(true)
      toast({
        title: "Model Generated",
        description: "PyTorch model code generated successfully",
      })
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: "Failed to generate model code",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
    }
  }, [nodes, edges, toast])

  const downloadCode = useCallback(() => {
    if (!generatedCode) return

    const blob = new Blob([generatedCode], { type: "text/python" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "neural_network_model.py"
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast({
      title: "Code Downloaded",
      description: "Model code saved as neural_network_model.py",
    })
  }, [generatedCode, toast])

  const copyCode = useCallback(async () => {
    if (!generatedCode) return

    try {
      await navigator.clipboard.writeText(generatedCode)
      setCopySuccess(true)
      toast({
        title: "Code Copied",
        description: "Generated code copied to clipboard",
      })
      setTimeout(() => setCopySuccess(false), 2000)
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy code to clipboard",
        variant: "destructive",
      })
    }
  }, [generatedCode, toast])

  const parsePyTorchCode = useCallback(
    (code: string) => {
      const errors: string[] = []
      const warnings: string[] = []
      const unsupportedModules: string[] = []
      const newNodes: Node[] = []
      const newEdges: Edge[] = []

      try {
        const classPatterns = [
          /class\s+(\w+)\s*\(\s*nn\.Module\s*\):/,
          /class\s+(\w+)\s*\(\s*torch\.nn\.Module\s*\):/,
          /class\s+(\w+)\s*\([^)]*nn\.Module[^)]*\):/,
          /class\s+(\w+)\s*\([^)]*torch\.nn\.Module[^)]*\):/,
        ]

        let className = ""
        let classFound = false

        for (const pattern of classPatterns) {
          const match = code.match(pattern)
          if (match) {
            className = match[1]
            classFound = true
            break
          }
        }

        if (!classFound) {
          errors.push(
            "No PyTorch model class found. Expected 'class YourModelName(nn.Module):' or similar inheritance from nn.Module",
          )
          return { nodes: [], edges: [], errors, warnings, unsupportedModules }
        }

        const initPattern = /def\s+__init__\s*\([^)]*\)\s*:([\s\S]*?)(?=\n\s*def\s+\w+|\n\s*class\s+\w+|$)/
        const initMatch = code.match(initPattern)

        if (!initMatch) {
          errors.push("No __init__ method found in the model class")
          return { nodes: [], edges: [], errors, warnings, unsupportedModules }
        }

        const initContent = initMatch[1]

        const layerPatterns = [
          /self\.(\w+)\s*=\s*nn\.(\w+)\s*\(([^)]*)\)/g,
          /self\.(\w+)\s*=\s*torch\.nn\.(\w+)\s*\(([^)]*)\)/g,
        ]

        const allLayerMatches: RegExpMatchArray[] = []

        for (const pattern of layerPatterns) {
          let match
          while ((match = pattern.exec(initContent)) !== null) {
            allLayerMatches.push(match)
          }
        }

        if (allLayerMatches.length === 0) {
          errors.push(
            "No PyTorch layers found in __init__ method. Make sure layers are defined as 'self.layer_name = nn.LayerType(...)'",
          )
          return { nodes: [], edges: [], errors, warnings, unsupportedModules }
        }

        const nodeTypeMap: Record<string, string> = {
          Linear: "linearNode",
          Conv1d: "conv1dNode",
          Conv2d: "conv2dNode",
          Conv3d: "conv3dNode",
          ConvTranspose1d: "convtranspose1dNode",
          ConvTranspose2d: "convtranspose2dNode",
          ConvTranspose3d: "convtranspose3dNode",
          BatchNorm1d: "batchnorm1dNode",
          BatchNorm2d: "batchnorm2dNode",
          BatchNorm3d: "batchnorm3dNode",
          LayerNorm: "layernormNode",
          GroupNorm: "groupnormNode",
          InstanceNorm1d: "instancenorm1dNode",
          InstanceNorm2d: "instancenorm2dNode",
          InstanceNorm3d: "instancenorm3dNode",
          ReLU: "reluNode",
          LeakyReLU: "leakyreluNode",
          ELU: "eluNode",
          GELU: "geluNode",
          SiLU: "siluNode",
          Mish: "mishNode",
          Hardswish: "hardswishNode",
          Hardsigmoid: "hardsigmoidNode",
          Tanh: "tanhNode",
          Sigmoid: "sigmoidNode",
          Softmax: "softmaxNode",
          LogSoftmax: "logsoftmaxNode",
          MaxPool1d: "maxpool1dNode",
          MaxPool2d: "maxpool2dNode",
          MaxPool3d: "maxpool3dNode",
          AvgPool1d: "avgpool1dNode",
          AvgPool2d: "avgpool2dNode",
          AvgPool3d: "avgpool3dNode",
          AdaptiveAvgPool1d: "adaptiveavgpool1dNode",
          AdaptiveAvgPool2d: "adaptiveavgpool2dNode",
          AdaptiveAvgPool3d: "adaptiveavgpool3dNode",
          AdaptiveMaxPool1d: "adaptivemaxpool1dNode",
          AdaptiveMaxPool2d: "adaptivemaxpool2dNode",
          Dropout: "dropoutNode",
          Dropout2d: "dropout2dNode",
          Dropout3d: "dropout3dNode",
          LSTM: "lstmNode",
          GRU: "gruNode",
          MultiheadAttention: "multiheadattentionNode",
          TransformerEncoderLayer: "transformerencoderlayerNode",
          TransformerDecoderLayer: "transformerdecoderlayerNode",
          Flatten: "flattenNode",
          Unflatten: "unflattenNode",
          Transpose: "transposeNode",
          Add: "addNode",
          Concatenate: "concatenateNode",
        }

        let yPosition = 100
        const layerMap = new Map<string, string>()

        allLayerMatches.forEach((match, index) => {
          const [, layerName, layerType, params] = match
          const nodeId = `${layerName}-${Date.now()}-${index}`
          layerMap.set(layerName, nodeId)

          const nodeType = nodeTypeMap[layerType]
          if (!nodeType) {
            unsupportedModules.push(layerType)
            warnings.push(`Unsupported layer type: ${layerType}. This layer will be skipped in visualization.`)
            return
          }

          const nodeData: any = { label: layerName }

          if (params && params.trim()) {
            try {
              const paramPairs = parseParameters(params)
              paramPairs.forEach((param) => {
                if (!param) return

                if (param.includes("=")) {
                  const [key, value] = param.split("=").map((s) => s.trim())
                  if (key && value) {
                    nodeData[key] = parseParameterValue(value)
                  }
                } else {
                  const numValue = Number.parseInt(param.trim())
                  if (!isNaN(numValue)) {
                    mapPositionalParameter(layerType, nodeData, numValue)
                  }
                }
              })
            } catch (paramError) {
              warnings.push(`Could not parse parameters for ${layerName}: ${params}`)
            }
          }

          const newNode: Node = {
            id: nodeId,
            type: nodeType,
            position: { x: 300, y: yPosition },
            data: nodeData,
            draggable: true,
            selectable: true,
            deletable: true,
          }

          newNodes.push(newNode)
          yPosition += 120
        })

        for (let i = 0; i < newNodes.length - 1; i++) {
          const edgeId = `e-${newNodes[i].id}-${newNodes[i + 1].id}`
          newEdges.push({
            id: edgeId,
            source: newNodes[i].id,
            target: newNodes[i + 1].id,
            type: "default",
            animated: false,
            deletable: true,
          })
        }

        if (newNodes.length > 0) {
          const inputNode: Node = {
            id: `input-${Date.now()}`,
            type: "inputNode",
            position: { x: 300, y: 20 },
            data: { channels: 3, height: 224, width: 224 },
            draggable: true,
            selectable: true,
            deletable: true,
          }

          newNodes.unshift(inputNode)

          const inputEdgeId = `e-${inputNode.id}-${newNodes[1].id}`
          newEdges.unshift({
            id: inputEdgeId,
            source: inputNode.id,
            target: newNodes[1].id,
            type: "default",
            animated: false,
            deletable: true,
          })
        }

        if (unsupportedModules.length > 0) {
          const uniqueUnsupported = [...new Set(unsupportedModules)]
          warnings.push(`Found ${uniqueUnsupported.length} unsupported module types: ${uniqueUnsupported.join(", ")}`)
          warnings.push("Consider requesting support for these modules in future updates.")
        }

        return { nodes: newNodes, edges: newEdges, errors, warnings, unsupportedModules }
      } catch (error) {
        errors.push(`Parsing error: ${error instanceof Error ? error.message : "Unknown error"}`)
        return { nodes: [], edges: [], errors, warnings, unsupportedModules }
      }
    },
    [],
  )

  const parseParameters = (params: string): string[] => {
    const result: string[] = []
    let current = ""
    let depth = 0
    let inQuotes = false
    let quoteChar = ""

    for (let i = 0; i < params.length; i++) {
      const char = params[i]

      if (!inQuotes && (char === "'" || char === '\"')) {
        inQuotes = true
        quoteChar = char
      } else if (inQuotes && char === quoteChar) {
        inQuotes = false
        quoteChar = ""
      } else if (!inQuotes && char === "(") {
        depth++
      } else if (!inQuotes && char === ")") {
        depth--
      } else if (!inQuotes && char === "," && depth === 0) {
        if (current.trim()) {
          result.push(current.trim())
        }
        current = ""
        continue
      }

      current += char
    }

    if (current.trim()) {
      result.push(current.trim())
    }

    return result
  }

  const parseParameterValue = (value: string): any => {
    const trimmedValue = value.trim()

    if (trimmedValue === "True") return true
    if (trimmedValue === "False") return false
    if (trimmedValue === "None") return null
    if (/^\d+$/.test(trimmedValue)) return Number.parseInt(trimmedValue)
    if (/^\d*\.\d+$/.test(trimmedValue)) return Number.parseFloat(trimmedValue)

    if (trimmedValue.startsWith("(") && trimmedValue.endsWith(")")) {
      let tupleContent = trimmedValue.slice(1, -1)
      if (tupleContent.endsWith(",")) {
        tupleContent = tupleContent.slice(0, -1)
      }
      const tupleValues = tupleContent.split(",").map((v) => {
        const tv = v.trim()
        if (tv === "") return null
        const num = Number.parseInt(tv)
        return isNaN(num) ? parseParameterValue(tv) : num
      })
      return tupleValues.length === 1 ? tupleValues[0] : tupleValues
    }

    if ((trimmedValue.startsWith("'") && trimmedValue.endsWith("'")) || (trimmedValue.startsWith('\"') && trimmedValue.endsWith('\"'))) {
      return trimmedValue.slice(1, -1)
    }

    return trimmedValue
  }

  const mapPositionalParameter = (layerType: string, nodeData: any, numValue: number): void => {
    const paramCount = Object.keys(nodeData).length - 1

    if (layerType === "Linear") {
      if (paramCount === 0) nodeData.in_features = numValue
      else if (paramCount === 1) nodeData.out_features = numValue
    } else if (layerType.includes("Conv")) {
      if (paramCount === 0) nodeData.in_channels = numValue
      else if (paramCount === 1) nodeData.out_channels = numValue
      else if (paramCount === 2) nodeData.kernel_size = numValue
    }
  }

  const handleCodeInput = useCallback(() => {
    const result = parsePyTorchCode(inputCode)

    if (result.errors.length > 0) {
      setParseErrors(result.errors)
      setParseWarnings([])
      setUnsupportedModules([])
      return
    }

    if (result.nodes.length === 0) {
      setParseErrors(["No valid layers found in the code"])
      setParseWarnings([])
      setUnsupportedModules([])
      return
    }

    takeSnapshot()
    setNodes(result.nodes)
    setEdges(result.edges)
    setParseErrors([])
    setParseWarnings(result.warnings)
    setUnsupportedModules(result.unsupportedModules)
    setShowCodeInputDialog(false)
    setInputCode("")

    if (result.nodes.length > 0) {
      const supportedCount = result.nodes.length - 1
      const totalLayers = supportedCount + result.unsupportedModules.length

      toast({
        title: "PyTorch Code Imported",
        description: `Successfully imported ${supportedCount} / ${totalLayers} layers. ${
          result.unsupportedModules.length > 0
            ? `${result.unsupportedModules.length} unsupported modules were skipped.`
            : "All modules are supported!"
        }`,
        duration: 5000,
      })
    }
  }, [inputCode, parsePyTorchCode, toast, setNodes, setEdges, takeSnapshot])

  const keyboardShortcuts = [
    { key: "Ctrl + S", description: "Save model" },
    { key: "Ctrl + O", description: "Open model" },
    { key: "Ctrl + G", description: "Generate PyTorch code" },
    { key: "Ctrl + R", description: "Reset canvas" },
    { key: "Delete / Backspace", description: "Delete selected nodes" },
  ]

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div className="flex items-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">{currentModelName || "Neural Network Designer"}</h1>
            <p className="text-sm text-muted-foreground">Build PyTorch models visually</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setShowHelpDialog(true)}>
            <HelpCircle className="h-4 w-4 mr-2" />
            Help
          </Button>
          <Button variant="outline" size="sm" onClick={() => setShowSaveDialog(true)}>
            <Save className="h-4 w-4 mr-2" />
            Save
          </Button>
          <Button variant="outline" size="sm" onClick={handleOpenLoadDialog}>
            <FolderOpen className="h-4 w-4 mr-2" />
            Open
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Network className="h-4 w-4 mr-2" />
                Load Example
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              {EXAMPLE_NETWORKS.map((example) => (
                <DropdownMenuItem key={example.name} onClick={() => loadExample(example)}>
                  <div className="flex flex-col">
                    <span className="font-medium">{example.name}</span>
                    <span className="text-xs text-muted-foreground">{example.description}</span>
                  </div>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <Button variant="outline" size="sm" onClick={undo} disabled={!canUndo}>
            <Undo2 className="h-4 w-4 mr-2" />
            Undo
          </Button>
          <Button variant="outline" size="sm" onClick={redo} disabled={!canRedo}>
            <Redo2 className="h-4 w-4 mr-2" />
            Redo
          </Button>
          <Button variant="outline" size="sm" onClick={resetCanvas}>
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button variant="outline" size="sm" onClick={analyzeCurrentModel} disabled={nodes.length === 0}>
            <BarChart3 className="h-4 w-4 mr-2" />
            Model Analysis
          </Button>
          <Button onClick={generateModel} disabled={isGenerating} className="flex items-center">
            {isGenerating ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Code className="h-4 w-4 mr-2" />}
            Generate PyTorch Code
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div className="w-64 bg-sidebar border-r border-sidebar-border">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
              <h2 className="font-semibold text-sidebar-foreground mb-3">Layer Library</h2>
              {/* Input Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Input</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("inputNode", { channels: 3, height: 28, width: 28 })}
                  >
                    <div className="flex items-center gap-2">
                      <Database className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Input</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("constantNode", { channels: 1, height: 1, width: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Box className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Constant</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Linear Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Linear</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("linearNode", { in_features: 128, out_features: 64 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-blue-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Linear</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Convolution Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Convolution</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("conv1dNode", { in_channels: 1, out_channels: 32, kernel_size: 3 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv1D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("conv2dNode", { in_channels: 3, out_channels: 32, kernel_size: 3 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("conv3dNode", { in_channels: 3, out_channels: 32, kernel_size: 3 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv3D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("depthwiseconv2dNode", { in_channels: 32, out_channels: 32, kernel_size: 3, groups: 32 })
                    }
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-orange-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">DepthwiseConv2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("separableconv2dNode", { in_channels: 32, out_channels: 64, kernel_size: 3 })
                    }
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">SeparableConv2D</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Transposed Convolution Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Transposed Convolution</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("convtranspose1dNode", { in_channels: 32, out_channels: 16, kernel_size: 3 })
                    }
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">ConvTranspose1D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("convtranspose2dNode", { in_channels: 32, out_channels: 16, kernel_size: 3 })
                    }
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">ConvTranspose2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("convtranspose3dNode", { in_channels: 32, out_channels: 16, kernel_size: 3 })
                    }
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">ConvTranspose3D</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Activation Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Activation</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("reluNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">ReLU</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("leakyreluNode", { negative_slope: 0.01 })}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">LeakyReLU</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("geluNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">GELU</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("siluNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">SiLU</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("mishNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Mish</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("hardswishNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Hardswish</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("hardsigmoidNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Hardsigmoid</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("sigmoidNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Sigmoid</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("tanhNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Tanh</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("softmaxNode", { dim: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Softmax</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Pooling Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Pooling</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("maxpool2dNode", { kernel_size: 2, stride: 2, padding: 0 })}
                  >
                    <div className="flex items-center gap-2">
                      <Shrink className="h-4 w-4 text-red-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">MaxPool2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("avgpool2dNode", { kernel_size: 2, stride: 2, padding: 0 })}
                  >
                    <div className="flex items-center gap-2">
                      <Shrink className="h-4 w-4 text-red-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">AvgPool2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("adaptiveavgpool2dNode", { output_size: [1, 1] })}
                  >
                    <div className="flex items-center gap-2">
                      <Shrink className="h-4 w-4 text-red-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">AdaptiveAvgPool2D</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Normalization Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Normalization</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("batchnorm1dNode", { num_features: 128 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">BatchNorm1D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("batchnorm2dNode", { num_features: 32 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">BatchNorm2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("layernormNode", { normalized_shape: [128] })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">LayerNorm</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("groupnormNode", { num_groups: 8, num_channels: 32 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">GroupNorm</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("instancenorm1dNode", { num_features: 128 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">InstanceNorm1D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("instancenorm2dNode", { num_features: 32 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">InstanceNorm2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("instancenorm3dNode", { num_features: 16 })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">InstanceNorm3D</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Regularization Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Regularization</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("dropoutNode", { p: 0.5 })}
                  >
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Dropout</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Utility Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Utility</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("flattenNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-gray-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Flatten</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("transposeNode", { dim0: 0, dim1: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <RotateCcw className="h-4 w-4 text-orange-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Transpose</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Advanced Operations Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Advanced Operations</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("addNode", { num_inputs: 2 })}
                  >
                    <div className="flex items-center gap-2">
                      <Plus className="h-4 w-4 text-orange-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Add (Skip Connection)</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("concatenateNode", { dim: 1, num_inputs: 2 })}
                  >
                    <div className="flex items-center gap-2">
                      <GitBranch className="h-4 w-4 text-indigo-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Concatenate</span>
                    </div>
                  </Card>
                </div>
              </div>

              {/* Recurrent Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Recurrent</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("lstmNode", { input_size: 128, hidden_size: 64 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">LSTM</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("gruNode", { input_size: 128, hidden_size: 64 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">GRU</span>
                    </div>
                  </Card>
                </div>
              </div>
              {/* Attention Section */}
              <div>
                <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">Attention</div>
                <div className="space-y-2">
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("multiheadattentionNode", { embed_dim: 128, num_heads: 8 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">MultiheadAttention</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("transformerencoderlayerNode", { d_model: 512, nhead: 8 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">TransformerEncoderLayer</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("transformerdecoderlayerNode", { d_model: 512, nhead: 8 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">TransformerDecoderLayer</span>
                    </div>
                  </Card>
                </div>
              </div>
            </div>
          </ScrollArea>
        </div>

        {/* Main Canvas Area */}
        <div className="flex-1 h-full w-full" ref={reactFlowWrapper}>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onNodeClick={onNodeClick}
              onPaneClick={() => setSelectedNode(null)}
              onEdgeClick={() => setSelectedNode(null)}
              nodeTypes={nodeTypes}
              className="h-full w-full"
              fitView
              deleteKeyCode={['Delete', 'Backspace']}
              tabIndex={0}
              onInit={(reactFlowInstance) => {
                reactFlowInstanceRef.current = reactFlowInstance
              }}
            >
              <Controls />
              <MiniMap />
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            </ReactFlow>
          </ReactFlowProvider>
        </div>

        {/* Right Panel: Properties & Live Validation */}
        <div className="w-80 bg-sidebar border-l border-sidebar-border flex flex-col">
          {/* Properties Panel */}
          <div ref={propertiesPanelRef} className="p-4 flex-shrink-0 overflow-y-auto">
            <h3 className="font-semibold text-sidebar-foreground mb-4">Properties</h3>
            {selectedNode ? (
              <div className="space-y-4">
                <div>
                  <Badge variant="secondary" className="mb-2">
                    {selectedNode.type}
                  </Badge>
                  <div className="text-sm text-sidebar-foreground/70">ID: {selectedNode.id}</div>
                </div>
                <Separator />
                <div className="space-y-3">
                  {selectedNode.type === "inputNode" && (
                    <>
                      <EditableNumberInput
                        label="Channels"
                        value={selectedNode.data.channels as number | undefined}
                        defaultValue={3}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { channels: value })}
                      />
                      <EditableNumberInput
                        label="Height"
                        value={selectedNode.data.height as number | undefined}
                        defaultValue={28}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { height: value })}
                      />
                      <EditableNumberInput
                        label="Width"
                        value={selectedNode.data.width as number | undefined}
                        defaultValue={28}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { width: value })}
                      />
                      <div className="p-2 bg-sidebar-accent/50 rounded text-xs text-sidebar-foreground/70">
                        Shape: {formatTensorShape(selectedNode.data.outputShape)}
                      </div>
                    </>
                  )}
                  {selectedNode.type === "constantNode" && (
                    <>
                      <EditableNumberInput
                        label="Channels"
                        value={selectedNode.data.channels as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { channels: value })}
                      />
                      <EditableNumberInput
                        label="Height"
                        value={selectedNode.data.height as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { height: value })}
                      />
                      <EditableNumberInput
                        label="Width"
                        value={selectedNode.data.width as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { width: value })}
                      />
                      <div className="p-2 bg-sidebar-accent/50 rounded text-xs text-sidebar-foreground/70">
                        Shape: {formatTensorShape(selectedNode.data.outputShape)}
                      </div>
                    </>
                  )}
                  {selectedNode.type === "linearNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Features"
                        value={selectedNode.data.in_features as number | undefined}
                        defaultValue={128}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_features: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Output Features"
                        value={selectedNode.data.out_features as number | undefined}
                        defaultValue={64}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { out_features: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "conv2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Channels"
                        value={selectedNode.data.in_channels as number | undefined}
                        defaultValue={3}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_channels: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Output Channels"
                        value={selectedNode.data.out_channels as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { out_channels: value })}
                      />
                      <EditableNumberInput
                        label="Kernel Size"
                        value={selectedNode.data.kernel_size as number | undefined}
                        defaultValue={3}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { kernel_size: value })}
                      />
                      <EditableNumberInput
                        label="Padding"
                        value={selectedNode.data.padding as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "separableconv2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Channels"
                        value={selectedNode.data.in_channels as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_channels: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Output Channels"
                        value={selectedNode.data.out_channels as number | undefined}
                        defaultValue={64}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { out_channels: value })}
                      />
                      <EditableNumberInput
                        label="Kernel Size"
                        value={selectedNode.data.kernel_size as number | undefined}
                        defaultValue={3}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { kernel_size: value })}
                      />
                      <EditableNumberInput
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <EditableNumberInput
                        label="Padding"
                        value={selectedNode.data.padding as number | undefined}
                        defaultValue={1}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding: value })}
                      />
                      <EditableNumberInput
                        label="Dilation"
                        value={selectedNode.data.dilation as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dilation: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "dropoutNode" && (
                    <>
                      <EditableNumberInput
                        label="Dropout Probability (p)"
                        value={selectedNode.data.p as number | undefined}
                        defaultValue={0.5}
                        min={0}
                        max={1}
                        step={0.01}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { p: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "batchnorm2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Number of Features"
                        value={selectedNode.data.num_features as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_features: value })}
                        disabled={isInputConnected}
                      />
                    </>
                  )}
                  {selectedNode.type === "convtranspose2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Channels"
                        value={selectedNode.data.in_channels as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_channels: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Output Channels"
                        value={selectedNode.data.out_channels as number | undefined}
                        defaultValue={16}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { out_channels: value })}
                      />
                      <EditableNumberInput
                        label="Kernel Size"
                        value={selectedNode.data.kernel_size as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { kernel_size: value })}
                      />
                      <EditableNumberInput
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <EditableNumberInput
                        label="Padding"
                        value={selectedNode.data.padding as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding: value })}
                      />
                      <EditableNumberInput
                        label="Output Padding"
                        value={selectedNode.data.output_padding as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { output_padding: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "maxpool2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Kernel Size"
                        value={selectedNode.data.kernel_size as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { kernel_size: value })}
                      />
                      <EditableNumberInput
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <EditableNumberInput
                        label="Padding"
                        value={selectedNode.data.padding as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "avgpool2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Kernel Size"
                        value={selectedNode.data.kernel_size as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { kernel_size: value })}
                      />
                      <EditableNumberInput
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <EditableNumberInput
                        label="Padding"
                        value={selectedNode.data.padding as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "addNode" && (
                    <>
                      <EditableNumberInput
                        label="Number of Inputs"
                        value={selectedNode.data.num_inputs as number | undefined}
                        defaultValue={2}
                        min={2}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_inputs: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "concatenateNode" && (
                    <>
                      <EditableNumberInput
                        label="Number of Inputs"
                        value={selectedNode.data.num_inputs as number | undefined}
                        defaultValue={2}
                        min={2}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_inputs: value })}
                      />
                      <div>
                        <label className="text-sm font-medium text-sidebar-foreground">Dimension (dim)</label>
                        <select
                          value={selectedNode.data.dim ?? 1}
                          onChange={(e) =>
                            updateNodeData(selectedNode.id, { dim: parseInt(e.target.value, 10) })
                          }
                          className="w-full px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500 bg-white text-black"
                        >
                          <option value={1}>1</option>
                          <option value={2}>2</option>
                          <option value={3}>3</option>
                        </select>
                        <p className="text-xs text-sidebar-foreground/70 mt-1">
                          Note: `dim=0` is for batch size, which is not applicable in this app.
                        </p>
                      </div>
                    </>
                  )}
                  {selectedNode.type === "adaptiveavgpool2dNode" && (
                    <>
                      <EditableNumberInput
                        label="Output Height"
                        value={
                          selectedNode.data.output_size && Array.isArray(selectedNode.data.output_size)
                            ? (selectedNode.data.output_size[0] as number | undefined) ?? 1
                            : 1
                        }
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) =>
                          updateNodeData(selectedNode.id, {
                            output_size: [
                              value,
                              selectedNode.data.output_size && Array.isArray(selectedNode.data.output_size)
                                ? (selectedNode.data.output_size[1] as number | undefined) ?? 1
                                : 1,
                            ],
                          })
                        }
                      />
                      <EditableNumberInput
                        label="Output Width"
                        value={
                          selectedNode.data.output_size && Array.isArray(selectedNode.data.output_size)
                            ? (selectedNode.data.output_size[1] as number | undefined) ?? 1
                            : 1
                        }
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) =>
                          updateNodeData(selectedNode.id, {
                            output_size: [
                              selectedNode.data.output_size && Array.isArray(selectedNode.data.output_size)
                                ? (selectedNode.data.output_size[0] as number | undefined) ?? 1
                                : 1,
                              value,
                            ],
                          })
                        }
                      />
                    </>
                  )}
                  {selectedNode.type === "layernormNode" && (
                    <>
                      <EditableNumberInput
                        label="Normalized Shape"
                        value={selectedNode.data.normalized_shape as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { normalized_shape: value })}
                      />
                      <EditableNumberInput
                        label="Epsilon (eps)"
                        value={selectedNode.data.eps as number | undefined}
                        defaultValue={1e-5}
                        min={0}
                        step={1e-6}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { eps: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "groupnormNode" && (
                    <>
                      <EditableNumberInput
                        label="Number of Groups"
                        value={selectedNode.data.num_groups as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_groups: value })}
                      />
                      <EditableNumberInput
                        label="Number of Channels"
                        value={selectedNode.data.num_channels as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_channels: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Epsilon (eps)"
                        value={selectedNode.data.eps as number | undefined}
                        defaultValue={1e-5}
                        min={0}
                        step={1e-6}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { eps: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "lstmNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Size"
                        value={selectedNode.data.input_size as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { input_size: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Hidden Size"
                        value={selectedNode.data.hidden_size as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { hidden_size: value })}
                      />
                      <EditableNumberInput
                        label="Number of Layers"
                        value={selectedNode.data.num_layers as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_layers: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "gruNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Size"
                        value={selectedNode.data.input_size as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { input_size: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Hidden Size"
                        value={selectedNode.data.hidden_size as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { hidden_size: value })}
                      />
                      <EditableNumberInput
                        label="Number of Layers"
                        value={selectedNode.data.num_layers as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_layers: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "multiheadattentionNode" && (
                    <>
                      <EditableNumberInput
                        label="Embedding Dimension"
                        value={selectedNode.data.embed_dim as number | undefined}
                        defaultValue={128}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { embed_dim: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Number of Heads"
                        value={selectedNode.data.num_heads as number | undefined}
                        defaultValue={8}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_heads: value })}
                      />
                      <EditableNumberInput
                        label="Dropout"
                        value={selectedNode.data.dropout as number | undefined}
                        defaultValue={0.0}
                        min={0}
                        max={1}
                        step={0.01}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dropout: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "transformerencoderlayerNode" && (
                    <>
                      <EditableNumberInput
                        label="D_model"
                        value={selectedNode.data.d_model as number | undefined}
                        defaultValue={512}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { d_model: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Nhead"
                        value={selectedNode.data.nhead as number | undefined}
                        defaultValue={8}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { nhead: value })}
                      />
                      <EditableNumberInput
                        label="Dim Feedforward"
                        value={selectedNode.data.dim_feedforward as number | undefined}
                        defaultValue={2048}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dim_feedforward: value })}
                      />
                      <EditableNumberInput
                        label="Dropout"
                        value={selectedNode.data.dropout as number | undefined}
                        defaultValue={0.1}
                        min={0}
                        max={1}
                        step={0.01}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dropout: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "transformerdecoderlayerNode" && (
                    <>
                      <EditableNumberInput
                        label="D_model"
                        value={selectedNode.data.d_model as number | undefined}
                        defaultValue={512}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { d_model: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Nhead"
                        value={selectedNode.data.nhead as number | undefined}
                        defaultValue={8}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { nhead: value })}
                      />
                      <EditableNumberInput
                        label="Dim Feedforward"
                        value={selectedNode.data.dim_feedforward as number | undefined}
                        defaultValue={2048}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dim_feedforward: value })}
                      />
                      <EditableNumberInput
                        label="Dropout"
                        value={selectedNode.data.dropout as number | undefined}
                        defaultValue={0.1}
                        min={0}
                        max={1}
                        step={0.01}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dropout: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "transposeNode" && (
                    <>
                      <EditableNumberInput
                        label="Dimension 0"
                        value={selectedNode.data.dim0 as number | undefined}
                        defaultValue={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dim0: value })}
                      />
                      <EditableNumberInput
                        label="Dimension 1"
                        value={selectedNode.data.dim1 as number | undefined}
                        defaultValue={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dim1: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "selectNode" && (
                    <>
                      <EditableNumberInput
                        label="Dimension"
                        value={selectedNode.data.dim as number | undefined}
                        defaultValue={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { dim: value })}
                      />
                      <EditableNumberInput
                        label="Index"
                        value={selectedNode.data.index as number | undefined}
                        defaultValue={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { index: value })}
                      />
                    </>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center text-sidebar-foreground/60 py-8">
                <Layers className="h-12 w-12 mx-auto mb-4 text-sidebar-foreground/30" />
                <p>Select a node to view its properties</p>
              </div>
            )}
          </div>

          {/* Live Validation Panel */}
          <div className="flex-grow flex flex-col mt-4 border-t border-sidebar-border pt-4 overflow-y-auto">
            <h3 className="font-semibold text-sidebar-foreground mb-2 px-4">Live Validation</h3>
            <ScrollArea className="flex-grow">
              <div className="space-y-2 px-4 pb-4">
                {liveValidationResults && liveValidationResults.errors.length > 0 && (
                  <div>
                    <h4 className="font-bold flex items-center text-red-500">
                      <AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />
                      Errors ({liveValidationResults.errors.length})
                    </h4>
                    <ul className="mt-1 space-y-1 text-xs text-red-400">
                      {liveValidationResults.errors.map((error, index) => (
                        <li key={`error-${index}`} className="flex">
                          <span className="mr-1">-</span>
                          <span>{error}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {liveValidationResults && liveValidationResults.warnings.length > 0 && (
                  <div className="mt-3">
                    <h4 className="font-bold flex items-center text-yellow-500">
                      <AlertTriangle className="h-4 w-4 mr-2 flex-shrink-0" />
                      Warnings ({liveValidationResults.warnings.length})
                    </h4>
                    <ul className="mt-1 space-y-1 text-xs text-yellow-400">
                      {liveValidationResults.warnings.map((warning, index) => (
                        <li key={`warning-${index}`} className="flex">
                          <span className="mr-1">-</span>
                          <span>{warning}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {liveValidationResults &&
                  liveValidationResults.errors.length === 0 &&
                  liveValidationResults.warnings.length === 0 && (
                    <div className="text-green-500 text-sm flex items-center">
                      <CheckCircle className="h-4 w-4 mr-2" />
                      <p>No issues found. Your model is looking good!</p>
                    </div>
                  )}
              </div>
            </ScrollArea>
          </div>
        </div>
      </div>

      {/* Code Generation Dialog */}
      <Dialog open={showCodeDialog} onOpenChange={setShowCodeDialog}>
        <DialogContent className="w-[65vw] h-[80vh] flex flex-col bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Generated PyTorch Code</DialogTitle>
            <DialogDescription>
              Here is the PyTorch code for your model. You can copy it or download it as a Python file.
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 relative my-4 min-h-0">
            <ScrollArea className="h-full rounded-md border">
              <pre className="p-4 font-mono text-sm whitespace-pre-wrap">{generatedCode}</pre>
            </ScrollArea>
            {generatedCode && (
              <div className="absolute top-3 right-3 flex gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={copyCode}
                  className={`transition-colors ${
                    copySuccess ? "bg-green-100 text-green-900" : ""
                  }`}
                >
                  {copySuccess ? (
                    "Copied!"
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5 mr-1.5" />
                      Copy
                    </>
                  )}
                </Button>
                <Button variant="secondary" size="sm" onClick={downloadCode}>
                  <Download className="h-3.5 w-3.5 mr-1.5" />
                  Download
                </Button>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="default" onClick={() => setShowCodeDialog(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showAnalysisPanel} onOpenChange={setShowAnalysisPanel}>
        <DialogContent className="max-w-4xl max-h-[90vh] w-[65vw] bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Model Analysis</DialogTitle>
          </DialogHeader>
          {modelAnalysis && (
            <div className="space-y-6">
              {/* Model Summary */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600 break-words">
                    {formatNumber(modelAnalysis.totalParameters)}
                  </div>
                  <div className="text-sm text-blue-800">Total Parameters</div>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600 break-words">
                    {formatNumber(modelAnalysis.totalFLOPs)}
                  </div>
                  <div className="text-sm text-green-800">FLOPs</div>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600 break-words">
                    {modelAnalysis.modelSizeMB.toFixed(2)} MB
                  </div>
                  <div className="text-sm text-purple-800">Model Size</div>
                </div>
                <div className="p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600 break-words">
                    {modelAnalysis.estimatedInferenceTimeMs.toFixed(2)} ms
                  </div>
                  <div className="text-sm text-orange-800">Est. Inference Time</div>
                </div>
              </div>

              {/* Layer-by-Layer Analysis */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Layer Analysis</h3>
                <ScrollArea className="h-64 w-full rounded-md border">
                  <div className="p-4">
                    <div className="space-y-2">
                      {modelAnalysis.layers.map((layer, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 bg-gray-50 rounded-lg min-w-0"
                        >
                          <div className="flex-1 min-w-0 mr-4">
                            <div className="font-medium text-sm truncate">{layer.name}</div>
                            <div className="text-xs text-gray-600 truncate">{layer.type}</div>
                          </div>
                          <div className="flex gap-4 text-xs flex-shrink-0">
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-gray-900">
                                {formatNumber(layer.parameters)}
                              </div>
                              <div className="text-gray-600">Params</div>
                            </div>
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-gray-900">{formatNumber(layer.flops)}</div>
                              <div className="text-gray-600">FLOPs</div>
                            </div>
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-gray-900">
                                {layer.memoryMB.toFixed(1)} MB
                              </div>
                              <div className="text-gray-600">Memory</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </ScrollArea>
              </div>

              {/* Performance Insights */}
              <div>
                <h3 className="text-lg font-semibold mb-3">Performance Insights</h3>
                <div className="space-y-2 text-sm">
                  {modelAnalysis.totalParameters > 1e6 && (
                    <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                      <div className="font-medium text-yellow-800">Large Model</div>
                      <div className="text-yellow-700">
                        This model has {formatNumber(modelAnalysis.totalParameters)} parameters. Consider using
                        techniques like pruning or quantization for deployment.
                      </div>
                    </div>
                  )}
                  {modelAnalysis.totalFLOPs > 1e9 && (
                    <div className="p-3 bg-orange-50 border border-orange-200 rounded-lg">
                      <div className="font-medium text-orange-800">High Computational Cost</div>
                      <div className="text-orange-700">
                        This model requires {formatNumber(modelAnalysis.totalFLOPs)} FLOPs. Consider optimizing for
                        faster inference.
                      </div>
                    </div>
                  )}
                  {modelAnalysis.memoryUsageMB > 1000 && (
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="font-medium text-red-800">High Memory Usage</div>
                      <div className="text-red-700">
                        This model uses approximately {modelAnalysis.memoryUsageMB.toFixed(0)} MB of memory during
                        inference.
                      </div>
                    </div>
                  )}
                  {modelAnalysis.totalParameters < 1e4 && (
                    <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="font-medium text-green-800">Lightweight Model</div>
                      <div className="text-green-700">
                        This model is lightweight with only {formatNumber(modelAnalysis.totalParameters)} parameters,
                        suitable for mobile deployment.
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          <div className="flex justify-end pt-4 border-t">
            <Button variant="default" onClick={() => setShowAnalysisPanel(false)}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={showValidationPanel} onOpenChange={setShowValidationPanel}>
        <DialogContent className="w-[65vw] h-[80vh] flex flex-col bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Model Validation Results</DialogTitle>
          </DialogHeader>
          <div className="flex-1 my-4">
            <ScrollArea className="h-full rounded-md border p-4">
              {validationResults && (validationResults.errors.length > 0 || validationResults.warnings.length > 0) ? (
                <div className="space-y-4">
                  {validationResults.errors.length > 0 && (
                    <div>
                      <h3 className="font-semibold text-red-600">Errors ({validationResults.errors.length})</h3>
                      <ul className="list-disc list-inside mt-2 space-y-1">
                        {validationResults.errors.map((error, index) => (
                          <li key={index} className="text-sm">
                            {error}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {validationResults.warnings.length > 0 && (
                    <div>
                      <h3 className="font-semibold text-yellow-600">Warnings ({validationResults.warnings.length})</h3>
                      <ul className="list-disc list-inside mt-2 space-y-1">
                        {validationResults.warnings.map((warning, index) => (
                          <li key={index} className="text-sm">
                            {warning}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full">
                  <p className="text-lg font-semibold">No issues found.</p>
                  <p className="text-sm text-gray-700">Your model is looking good!</p>
                </div>
              )}
            </ScrollArea>
          </div>
          <div className="flex justify-end">
            <Button variant="default" onClick={() => setShowValidationPanel(false)}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent className="w-[50vw] bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Save Model</DialogTitle>
            <DialogDescription>
              Enter a name to save the model in the browser, or export it as a .json file.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <Input
              id="model-name"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="My-CNN-Model"
              className="bg-white text-black"
            />
          </div>
          <DialogFooter>
            <Button variant="secondary" onClick={handleExportModel}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
            <Button variant="default" onClick={handleSaveModel}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showLoadDialog} onOpenChange={setShowLoadDialog}>
        <DialogContent className="w-[50vw] bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Open Model</DialogTitle>
            <DialogDescription>
              Select a saved model to load it onto the canvas, or import a model from a .json file.
            </DialogDescription>
          </DialogHeader>
          <div className="my-4">
            <ScrollArea className="h-72 rounded-md border">
              <div className="p-4 space-y-2">
                {savedModels.length > 0 ? (
                  savedModels.map((model) => (
                    <div
                      key={model.key}
                      className="flex items-center justify-between rounded-lg border p-3 transition-all hover:bg-gray-100"
                    >
                      <div>
                        <div className="font-semibold">{model.name}</div>
                        <div className="text-xs text-gray-600">Saved: {new Date(model.timestamp).toLocaleString()}</div>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" onClick={() => handleLoadModel(model.key)}>
                          Load
                        </Button>
                        <Button variant="destructive" size="sm" onClick={() => handleDeleteModel(model.key)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-gray-500 py-10">No saved models found.</div>
                )}
              </div>
            </ScrollArea>
          </div>
          <DialogFooter>
            <Button asChild variant="secondary">
              <label htmlFor="import-model-input" className="cursor-pointer flex items-center">
                <FileUp className="h-4 w-4 mr-2" />
                Import from File
              </label>
            </Button>
            <input
              type="file"
              id="import-model-input"
              accept=".json"
              onChange={handleImportModel}
              className="hidden"
            />
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showCodeInputDialog} onOpenChange={setShowCodeInputDialog}>
        <DialogContent className="w-[80vw] h-[80vh] flex flex-col bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Input PyTorch Code</DialogTitle>
            <DialogDescription>
              Paste your PyTorch model code below and we'll recreate it visually in the canvas
            </DialogDescription>
          </DialogHeader>
          <div className="flex-1 flex flex-col gap-4 py-4 overflow-y-auto">
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">PyTorch Model Code:</label>
              <textarea
                value={inputCode}
                onChange={(e) => setInputCode(e.target.value)}
                placeholder={`import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 112 * 112, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x`}
                className="w-full h-full p-3 border rounded-md font-mono text-sm resize-none"
              ></textarea>
            </div>
            {parseErrors.length > 0 && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                <strong className="font-bold">Parsing Errors:</strong>
                <ul className="mt-1 list-disc list-inside">
                  {parseErrors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              </div>
            )}
            {parseWarnings.length > 0 && (
              <div
                className="bg-yellow-100 border border-yellow-400 text-yellow-700 px-4 py-3 rounded relative"
                role="alert"
              >
                <strong className="font-bold">Parsing Warnings:</strong>
                <ul className="mt-1 list-disc list-inside">
                  {parseWarnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            {unsupportedModules.length > 0 && (
              <div
                className="bg-orange-100 border border-orange-400 text-orange-700 px-4 py-3 rounded relative"
                role="alert"
              >
                <strong className="font-bold">Unsupported Modules:</strong>
                <ul className="mt-1 list-disc list-inside">
                  {unsupportedModules.map((module, index) => (
                    <li key={index}>{module}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCodeInputDialog(false)}>
              Close
            </Button>
            <Button onClick={handleCodeInput}>
              <Zap className="h-4 w-4 mr-2" />
              Parse Code
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showHelpDialog} onOpenChange={setShowHelpDialog}>
        <DialogContent className="max-w-4xl max-h-[90vh] w-[65vw] bg-gray-200 text-gray-900">
          <DialogHeader>
            <DialogTitle>Neural Network Designer - Help Guide</DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[70vh] pr-4">
            <div className="space-y-6">
              {/* Getting Started */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🚀 Getting Started</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    Welcome to the Neural Network Designer! This tool helps you build PyTorch models visually by
                    connecting blocks on a canvas.
                  </p>
                </div>
              </div>

              {/* Adding Blocks */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📦 Adding Blocks</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Browse the left sidebar to find different layer types (Input, Linear,
                    Convolution, etc.)
                  </p>
                  <p>
                    <strong>2.</strong> Click on any block type to add it to the canvas
                  </p>
                  <p>
                    <strong>3.</strong> Blocks will appear in the center canvas area
                  </p>
                  <p>
                    <strong>4.</strong> Each block shows its input/output tensor shapes automatically
                  </p>
                </div>
              </div>

              {/* Connecting Blocks */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🔗 Connecting Blocks</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Drag from the output handle (right side) of one block
                  </p>
                  <p>
                    <strong>2.</strong> Drop onto the input handle (left side) of another block
                  </p>
                  <p>
                    <strong>3.</strong> Connections show the data flow through your network
                  </p>
                  <p>
                    <strong>4.</strong> Tensor shapes update automatically when you make connections
                  </p>
                </div>
              </div>

              {/* Configuring Parameters */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">⚙️ Configuring Parameters</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Click on any block to select it
                  </p>
                  <p>
                    <strong>2.</strong> The right sidebar shows editable parameters for that block
                  </p>
                  <p>
                    <strong>3.</strong> Type new values and press Enter to update
                  </p>
                  <p>
                    <strong>4.</strong> Tensor shapes recalculate automatically when parameters change
                  </p>
                </div>
              </div>

              {/* Deleting Blocks */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🗑️ Deleting Blocks</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Click on a block to select it (shows selection border)
                  </p>
                  <p>
                    <strong>2.</strong> Press the <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Delete</kbd> or{" "}
                    <kbd className="px-2 py-1 bg-gray-100 rounded text-xs">Backspace</kbd> key
                  </p>
                  <p>
                    <strong>3.</strong> The block and its connections will be removed
                  </p>
                </div>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">⌨️ Keyboard Shortcuts</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  {keyboardShortcuts.map((shortcut, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <p>{shortcut.description}</p>
                      <kbd className="px-2 py-1 bg-gray-300 rounded text-xs font-semibold">{shortcut.key}</kbd>
                    </div>
                  ))}
                </div>
              </div>

              {/* Saving, Loading, Importing & Exporting */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">💾 Saving, Loading, Importing & Exporting</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>To Save:</strong> Click the "Save" button, enter a name, and save it to your browser's local storage.
                  </p>
                  <p>
                    <strong>To Load:</strong> Click the "Open" button and select a previously saved model from the list.
                  </p>
                  <p>
                    <strong>To Export:</strong> Click the "Save" button, enter a name, and click "Export" to download your model as a `.json` file.
                  </p>
                  <p>
                    <strong>To Import:</strong> Click the "Open" button, then "Import from File" to load a `.json` model from your computer.
                  </p>
                </div>
              </div>

              {/* Using Examples */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📚 Loading Examples</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Click the "Load Example" button in the header
                  </p>
                  <p>
                    <strong>2.</strong> Choose from pre-built architectures like LeNet-5, ResNet, U-Net, YOLO, etc.
                  </p>
                  <p>
                    <strong>3.</strong> Examples load with proper connections and parameters
                  </p>
                  <p>
                    <strong>4.</strong> Use "Reset" to clear the canvas and start fresh
                  </p>
                </div>
              </div>

              {/* Generating Code */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🐍 Generating PyTorch Code</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Click "Generate PyTorch Code" when your network is ready
                  </p>
                  <p>
                    <strong>2.</strong> Review the generated Python code in the dialog
                  </p>
                  <p>
                    <strong>3.</strong> Use "Copy Code" to copy to clipboard
                  </p>
                  <p>
                    <strong>4.</strong> Use "Download Code" to save as a .py file
                  </p>
                </div>
              </div>

              {/* Model Analysis */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📊 Model Analysis</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    <strong>1.</strong> Click "Analyze Model" to get detailed insights
                  </p>
                  <p>
                    <strong>2.</strong> View parameter counts, FLOPs, and memory usage
                  </p>
                  <p>
                    <strong>3.</strong> See layer-by-layer analysis and performance recommendations
                  </p>
                  <p>
                    <strong>4.</strong> Use insights to optimize your architecture
                  </p>
                </div>
              </div>

              {/* Tips */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">💡 Pro Tips</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>• Always start with an Input block to define your data dimensions</p>
                  <p>• Watch tensor shapes to ensure compatibility between layers</p>
                  <p>• Use skip connections (Add/Concatenate blocks) for advanced architectures</p>
                  <p>• Load examples to learn common architectural patterns</p>
                  <p>• Analyze your model before generating code to catch issues early</p>
                </div>
              </div>

              {/* Available Blocks */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🧱 Available Block Types</h3>
                <div className="grid grid-cols-2 gap-4 text-sm text-gray-700">
                  <div>
                    <p>
                      <strong>Basic:</strong> Input, Linear, Dropout
                    </p>
                    <p>
                      <strong>Convolution:</strong> Conv1D/2D/3D, DepthwiseConv2D
                    </p>
                    <p>
                      <strong>Pooling:</strong> MaxPool, AvgPool, AdaptivePool
                    </p>
                    <p>
                      <strong>Activation:</strong> ReLU, GELU, SiLU, Sigmoid, Tanh
                    </p>
                  </div>
                  <div>
                    <p>
                      <strong>Normalization:</strong> BatchNorm, LayerNorm, GroupNorm
                    </p>
                    <p>
                      <strong>Recurrent:</strong> LSTM, GRU
                    </p>
                    <p>
                      <strong>Attention:</strong> MultiheadAttention, Transformer
                    </p>
                    <p>
                      <strong>Operations:</strong> Add, Concatenate, Flatten
                    </p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📧 Feedback & Contact</h3>
                <div className="space-y-2 text-sm text-gray-700">
                  <p>
                    Have suggestions, found a bug, or want to request new features? Send your feedback to:{" "}
                    <strong>pmquang87@icloud.com</strong>
                  </p>
                  <p>Thank you for using Neural Network Designer and for your valuable feedback!</p>
                </div>
              </div>
            </div>
          </ScrollArea>
          <div className="flex justify-end pt-4 border-t">
            <Button variant="default" onClick={() => setShowHelpDialog(false)}>
              Got it!
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
