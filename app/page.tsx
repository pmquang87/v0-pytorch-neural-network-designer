'use client'

import type React from "react"

import { useState, useCallback, useEffect, useMemo, useRef } from "react"
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
  type NodeChange,
  type EdgeChange,
  BackgroundVariant,
  Handle,
  Position,
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
  Hash,
  Loader2,
  Download,
  Copy,
  Zap,
  Shrink,
  Eye,
  Plus,
  X,
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
  ArrowUpRight,
  ArrowDownLeft,
  Share2,
} from "lucide-react"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { ResizableBox } from "react-resizable"
import "react-resizable/css/styles.css"

import { EXAMPLE_NETWORKS_METADATA, type ExampleNetworkMetadata } from "@/lib/example-networks"
import {
  calculateOutputShape as calcOutputShape,
  formatTensorShape,
  type TensorShape,
} from "@/lib/tensor-shape-calculator"

import { analyzeModel, formatNumber, type ModelAnalysis } from "@/lib/model-analyzer"
import { useAutoSave, StorageUtils } from "@/lib/auto-save"
import { useModelValidation } from "@/lib/model-validator"
import { useUndoRedo } from "@/lib/undo-redo"
import { useKeyboardShortcuts } from "@/lib/keyboard-shortcuts"
import { parsePyTorchModel } from "@/lib/pytorch-parser"
import { ModelGenerator, type TrainingOptions } from "@/lib/model-generator"
import { buildShareUrl, readGraphFromHash } from "@/lib/share"
import { formatModelSummary } from "@/lib/model-summary"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { EditableNumberInput } from "@/components/ui/EditableNumberInput"
import { ThemeToggle } from "@/components/theme-toggle"
import { ErrorBoundary } from "@/components/ErrorBoundary"

// Custom Node Components
import { InputNode } from "@/components/nodes/InputNode"
import { ConstantNode } from "@/components/nodes/ConstantNode"
import { MaxPool1DNode } from "@/components/nodes/MaxPool1DNode"
import { MaxPool3DNode } from "@/components/nodes/MaxPool3DNode"
import { BatchNorm3DNode } from "@/components/nodes/BatchNorm3DNode"
import { FractionalMaxPool2DNode } from "@/components/nodes/FractionalMaxPool2DNode"
import { LPPool2DNode } from "@/components/nodes/LPPool2DNode"
import { AdaptiveMaxPool1DNode } from "@/components/nodes/AdaptiveMaxPool1DNode"
import { LinearNode } from "@/components/nodes/LinearNode"
import { EmbeddingNode } from "@/components/nodes/EmbeddingNode"
import { TimeDistributedLinearNode } from "@/components/nodes/TimeDistributedLinearNode"
import { Conv2DNode } from "@/components/nodes/Conv2DNode"
import { ReLUNode } from "@/components/nodes/ReLUNode"
import { SigmoidNode } from "@/components/nodes/SigmoidNode"
import { TanhNode } from "@/components/nodes/TanhNode"
import { SoftmaxNode } from "@/components/nodes/SoftmaxNode"
import { LeakyReLUNode } from "@/components/nodes/LeakyReLUNode"
import { DropoutNode } from "@/components/nodes/DropoutNode"
import { FlattenNode } from "@/components/nodes/FlattenNode"
import { MaxPool2DNode } from "@/components/nodes/MaxPool2DNode"
import { AvgPool2DNode } from "@/components/nodes/AvgPool2DNode"
import { AdaptiveAvgPool2DNode } from "@/components/nodes/AdaptiveAvgPool2DNode"
import { BatchNorm1DNode } from "@/components/nodes/BatchNorm1DNode"
import { BatchNorm2DNode } from "@/components/nodes/BatchNorm2DNode"
import { LayerNormNode } from "@/components/nodes/LayerNormNode"
import { RMSNormNode } from "@/components/nodes/RMSNormNode"
import { MoENode } from "@/components/nodes/MoENode"
import { GroupNormNode } from "@/components/nodes/GroupNormNode"
import { ConcatenateNode } from "@/components/nodes/ConcatenateNode"
import { AddNode } from "@/components/nodes/AddNode"
import { MultiplyNode } from "@/components/nodes/MultiplyNode"
import { LSTMNode } from "@/components/nodes/LSTMNode"
import { GRUNode } from "@/components/nodes/GRUNode"
import { RNNNode } from "@/components/nodes/RNNNode"
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
import { ScaledDotProductAttentionNode } from "@/components/nodes/ScaledDotProductAttentionNode"
import { TransformerEncoderLayerNode } from "@/components/nodes/TransformerEncoderLayerNode"
import { TransformerDecoderLayerNode } from "@/components/nodes/TransformerDecoderLayerNode"
import { TransposeNode } from "@/components/nodes/TransposeNode"
import { SelectNode } from "@/components/nodes/SelectNode"
import { OutputNode } from "@/components/nodes/OutputNode"
import { ReshapeNode } from "@/components/nodes/ReshapeNode"
import { ChunkNode } from "@/components/nodes/ChunkNode"
import { SsmNode } from "@/components/nodes/SsmNode"
import { MBConvNode } from "@/components/nodes/MBConvNode"
import { NoiseNode } from "@/components/nodes/NoiseNode"
import { AdaptiveInstanceNormNode } from "@/components/nodes/AdaptiveInstanceNormNode"
import { UpsampleNode } from "@/components/nodes/UpsampleNode"
import { DownsampleNode } from "@/components/nodes/DownsampleNode"
import { InvertedResidualBlockNode } from "@/components/nodes/InvertedResidualBlockNode"
import { SEBlockNode } from "@/components/nodes/SEBlockNode";
import { SEBottleneckNode } from "@/components/nodes/SEBottleneckNode";

// Placeholder for conceptual nodes in examples
const DefaultNode = ({ data }: { data: { label: string } }) => (
  <div style={{
    padding: '10px',
    border: '1px solid #91d5ff',
    borderRadius: '5px',
    background: '#e6f7ff',
    textAlign: 'center'
  }}>
    <Handle type="target" position={Position.Left} />
    <p style={{ margin: 0, fontSize: '12px', fontWeight: 'bold', color: '#1890ff' }}>{data.label || 'Node'}</p>
    <Handle type="source" position={Position.Right} />
  </div>
);

const ContentLossNode = () => (
    <div style={{
        padding: '10px',
        border: '1px solid #ff4d4d',
        borderRadius: '5px',
        background: '#fff0f0',
        textAlign: 'center'
    }}>
        <Handle type="target" id="a" position={Position.Left} style={{ top: '30%' }} />
        <Handle type="target" id="b" position={Position.Left} style={{ top: '70%' }} />
        <p style={{ margin: 0, fontSize: '12px', color: '#d4380d' }}>Content Loss</p>
        <Handle type="source" position={Position.Right} />
    </div>
);

const StyleLossNode = () => (
    <div style={{
        padding: '10px',
        border: '1px solid #4d94ff',
        borderRadius: '5px',
        background: '#f0f5ff',
        textAlign: 'center'
    }}>
        <Handle type="target" id="a" position={Position.Left} style={{ top: '30%' }} />
        <Handle type="target" id="b" position={Position.Left} style={{ top: '70%' }} />
        <p style={{ margin: 0, fontSize: '12px', color: '#0052cc' }}>Style Loss</p>
        <Handle type="source" position={Position.Right} />
    </div>
);

const ParameterNode = ({ data }: { data: { label: string, shape: number[] } }) => (
    <div style={{
        padding: '10px',
        border: '1px solid #d3adf7',
        borderRadius: '5px',
        background: '#f9f0ff',
        textAlign: 'center'
    }}>
        <p style={{ margin: 0, fontSize: '12px', fontWeight: 'bold', color: '#531dab' }}>Parameter</p>
        <p style={{ margin: 0, fontSize: '11px', color: '#531dab' }}>{data.label || ''}</p>
        {data.shape && <p style={{ margin: 0, fontSize: '10px', color: '#722ed1' }}>{`Shape: [${data.shape.join(', ')}]`}</p>}
        <Handle type="source" position={Position.Right} />
    </div>
);

const initialNodes: Node[] = [
  {
    id: "inputNode_1",
    type: "inputNode",
    position: { x: 100, y: 100 },
    data: { channels: 3, height: 28, width: 28 },
  },
]

const initialEdges: Edge[] = []

const nodeTypes: NodeTypes = {
  inputNode: InputNode,
  constantNode: ConstantNode,
  parameterNode: ParameterNode,
  linearNode: LinearNode,
  embeddingNode: EmbeddingNode,
  timeDistributedLinearNode: TimeDistributedLinearNode,
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
  maxpool1dNode: MaxPool1DNode,
  maxpool2dNode: MaxPool2DNode,
  maxpool3dNode: MaxPool3DNode,
  avgpool2dNode: AvgPool2DNode,
  adaptiveavgpool2dNode: AdaptiveAvgPool2DNode,
  adaptivemaxpool1dNode: AdaptiveMaxPool1DNode,
  fractionalmaxpool2dNode: FractionalMaxPool2DNode,
  lppool2dNode: LPPool2DNode,
  batchnorm1dNode: BatchNorm1DNode,
  batchnorm2dNode: BatchNorm2DNode,
  batchnorm3dNode: BatchNorm3DNode,
  layernormNode: LayerNormNode,
  rmsnormNode: RMSNormNode,
  moeNode: MoENode,
  groupnormNode: GroupNormNode,
  instancenorm1dNode: InstanceNorm1DNode,
  instancenorm2dNode: InstanceNorm2DNode,
  instancenorm3dNode: InstanceNorm3DNode,
  concatenateNode: ConcatenateNode,
  addNode: AddNode,
  multiplyNode: MultiplyNode,
  lstmNode: LSTMNode,
  gruNode: GRUNode,
  rnnNode: RNNNode,
  multiheadattentionNode: MultiheadAttentionNode,
  scaledDotProductAttentionNode: ScaledDotProductAttentionNode,
  transformerencoderlayerNode: TransformerEncoderLayerNode,
  transformerdecoderlayerNode: TransformerDecoderLayerNode,
  transposeNode: TransposeNode,
  selectNode: SelectNode,
  outputNode: OutputNode,
  reshapeNode: ReshapeNode,
  chunkNode: ChunkNode,
  ssmNode: SsmNode,
  mbconvNode: MBConvNode,
  noiseNode: NoiseNode,
  adaptiveInstanceNormNode: AdaptiveInstanceNormNode,
  upsampleNode: UpsampleNode,
  downsampleNode: DownsampleNode,
  defaultNode: DefaultNode,
  contentLossNode: ContentLossNode,
  styleLossNode: StyleLossNode,
  invertedResidualBlockNode: InvertedResidualBlockNode,
  seBlockNode: SEBlockNode,
  seBottleneckNode: SEBottleneckNode,
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
    resetHistory,
    canUndo,
    canRedo,
  } = useUndoRedo(initialNodes, initialEdges)
  const [selectedNode, setSelectedNode] = useState<Node | null>(null)
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const removals = changes.filter((c: NodeChange) => c.type === "remove")
      if (removals.length > 0) {
        takeSnapshot()
        if (selectedNode && removals.some((r) => "id" in r && r.id === selectedNode.id)) {
          setSelectedNode(null)
        }
      }
      setNodes((nds) => applyNodeChanges(changes, nds))
    },
    [setNodes, takeSnapshot, selectedNode, setSelectedNode],
  )
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      if (changes.some((c: EdgeChange) => c.type === "remove")) {
        takeSnapshot()
      }
      setEdges((eds) => applyEdgeChanges(changes, eds))
    },
    [setEdges, takeSnapshot],
  )
  const [showCodeDialog, setShowCodeDialog] = useState(false)
  const [generatedCode, setGeneratedCode] = useState("")
  const [isGenerating, setIsGenerating] = useState(false)
  // Optional training-loop scaffold controls for the Generate Code dialog.
  const [includeTraining, setIncludeTraining] = useState(false)
  const [trainingOptimizer, setTrainingOptimizer] = useState<TrainingOptions["optimizer"]>("adam")
  const [trainingLoss, setTrainingLoss] = useState<TrainingOptions["loss"]>("crossentropy")
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
  const [editingNodeId, setEditingNodeId] = useState("")

  const reactFlowInstanceRef = useRef<any>(null)
  const autoSave = useAutoSave()
  const { validateModel } = useModelValidation()

  const isInputConnected = selectedNode ? edges.some((edge) => edge.target === selectedNode.id) : false

  // Live total-parameter count shown in the header
  const totalParameters = useMemo(() => {
    try {
      return analyzeModel(nodes, edges).totalParameters
    } catch {
      return 0
    }
  }, [nodes, edges])

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
    // A model shared via URL hash (#model=...) takes precedence over any
    // auto-saved session. Uses the same load path as import/load: seed the
    // graph + undo history, clear selection, fit the view.
    if (typeof window !== "undefined") {
      const sharedGraph = readGraphFromHash(window.location.hash)
      if (sharedGraph && Array.isArray(sharedGraph.nodes) && sharedGraph.nodes.length > 0) {
        resetHistory(sharedGraph.nodes as Node[], sharedGraph.edges as Edge[])
        setSelectedNode(null)
        toast({
          title: "Loaded shared model",
          description: "A model shared via link has been loaded onto the canvas.",
        })
        // Drop the hash so a refresh doesn't reload the shared model over edits.
        window.history.replaceState(null, "", window.location.pathname + window.location.search)
        setTimeout(() => {
          reactFlowInstanceRef.current?.fitView({ padding: 0.1, duration: 800 })
        }, 100)
        return
      }
    }

    if (autoSave.hasSavedData()) {
      const loadedState = autoSave.load()
      if (loadedState) {
        resetHistory(loadedState.nodes, loadedState.edges)
        toast({
          title: "Session Restored",
          description: "Your previous session has been loaded.",
        })
      }
    }
  }, [])

  useEffect(() => {
    if (selectedNode) {
      setEditingNodeId(selectedNode.id)
    }
  }, [selectedNode])

  useEffect(() => {
    const results = validateModel(nodes, edges)
    setLiveValidationResults(results)
  }, [nodes, edges, validateModel])

  const propagateTensorShapes = useCallback(() => {
    if (isUpdatingShapes.current) return;
    isUpdatingShapes.current = true;

    setNodes((currentNodes) => {
      const newNodes = structuredClone(currentNodes);
      const nodeMap = new Map(newNodes.map((node) => [node.id, node]));

      const visited = new Set<string>();
      const processing = new Set<string>();
      const sorted: string[] = [];

      const visit = (nodeId: string) => {
        if (processing.has(nodeId)) {
          console.warn("Cycle detected in graph, aborting shape propagation.");
          return;
        }
        if (visited.has(nodeId)) return;

        processing.add(nodeId);
        const outgoingEdges = edges.filter((edge) => edge.source === nodeId);
        for (const edge of outgoingEdges) {
          if (nodeMap.has(edge.target)) {
            visit(edge.target);
          }
        }
        processing.delete(nodeId);
        visited.add(nodeId);
        sorted.unshift(nodeId);
      };

      for (const node of newNodes) {
        if (!visited.has(node.id)) {
          visit(node.id);
        }
      }

      for (const nodeId of sorted) {
        const node = nodeMap.get(nodeId);
        if (!node) continue;

        if (node.type === "parameterNode") {
          const shapeArray = node.data.shape as number[];
          let outputShape: TensorShape = {};
          if (shapeArray) {
            if (shapeArray.length === 3) { // [batch, seq_len, features]
              outputShape = {
                sequence: shapeArray[1],
                features: shapeArray[2],
              };
            } else if (shapeArray.length === 2) { // [seq_len, features]
              outputShape = {
                sequence: shapeArray[0],
                features: shapeArray[1],
              };
            } else if (shapeArray.length === 1) { // [features]
              outputShape = {
                features: shapeArray[0],
              };
            }
          }
          node.data.outputShape = outputShape;
          node.data.inputShape = {}; // Parameters don't have input
          continue;
        }

        if (node.type === "inputNode" || node.type === "constantNode") {
          const isInput = node.type === "inputNode";
          const shape: TensorShape = {
            channels: (node.data as any).channels ?? (isInput ? 3 : 1),
            height: (node.data as any).height ?? (isInput ? 28 : 1),
            width: (node.data as any).width ?? (isInput ? 28 : 1),
          };
          node.data.inputShape = shape;
          node.data.outputShape = shape;
          continue;
        }

        const inputEdges = edges.filter((edge) => edge.target === nodeId);
        let allInputShapes: (TensorShape | undefined)[] = [];

        if (node.type === "concatenateNode" || node.type === "addNode" || node.type === "multiplyNode") {
          const connectedHandles = inputEdges
            .map((edge) => edge.targetHandle)
            .filter((handle): handle is string => handle !== null && handle !== undefined)
            .map((handle) => parseInt(handle.replace("input", ""), 10));
    
          const highestConnectedHandle = connectedHandles.length > 0 ? Math.max(...connectedHandles) : 0;
          const numConnected = inputEdges.length;
          const numInputs = Math.max(2, numConnected + 1, highestConnectedHandle);
    
          for (let i = 1; i <= numInputs; i++) {
            const handleId = `input${i}`;
            const edge = inputEdges.find((e) => e.targetHandle === handleId);
            if (edge) {
              const sourceNode = nodeMap.get(edge.source);
              allInputShapes.push(sourceNode?.data.outputShape);
            } else {
              allInputShapes.push(undefined);
            }
          }
        } else {
          allInputShapes = inputEdges.map((edge) => {
            const sourceNode = nodeMap.get(edge.source);
            return sourceNode?.data.outputShape;
          });
        }

        const cleanInputShapes = allInputShapes.filter((s): s is TensorShape => !!s);
        const data = node.data;
        const firstInputShape = cleanInputShapes[0];

        if (firstInputShape && firstInputShape.channels) {
          if (
            node.type === "conv1dNode" ||
            node.type === "conv2dNode" ||
            node.type === "conv3dNode" ||
            node.type === "convtranspose1dNode" ||
            node.type === "convtranspose2dNode" ||
            node.type === "convtranspose3dNode" ||
            node.type === "depthwiseconv2dNode" ||
            node.type === "separableconv2dNode" ||
            node.type === "mbconvNode" ||
            node.type === "invertedResidualBlockNode" ||
            node.type === "seBlockNode"
          ) {
            data.in_channels = firstInputShape.channels;
          } else if (node.type === "seBottleneckNode") {
            data.in_planes = firstInputShape.channels;
          } else if (
            node.type === "batchnorm1dNode" ||
            node.type === "batchnorm2dNode" ||
            node.type === "instancenorm1dNode" ||
            node.type === "instancenorm2dNode" ||
            node.type === "instancenorm3dNode"
          ) {
            data.num_features = firstInputShape.channels;
          } else if (node.type === "groupnormNode") {
            data.num_channels = firstInputShape.channels;
          }
        }

        if (node.type === "linearNode" && firstInputShape) {
            // nn.Linear applies to the LAST dimension only: (*, H_in) -> (*, H_out),
            // so in_features must track the input's last dimension — not the
            // flattened product of all dimensions (see issue #5)
            const linearDimOrder: (keyof TensorShape)[] = [
                "features", "length", "sequence", "width", "height", "depth", "channels",
            ];
            const lastDimKey = linearDimOrder.find((d) => (firstInputShape as any)[d] !== undefined);
            const lastDimValue = lastDimKey ? (firstInputShape as any)[lastDimKey] : undefined;
            if (typeof lastDimValue === "number") {
                data.in_features = lastDimValue;
            }
        }

        const outputShape = calcOutputShape(node.type || "", cleanInputShapes, data);

        if (node.type === "concatenateNode" || node.type === "addNode" || node.type === "multiplyNode") {
          node.data.inputShape = allInputShapes;
          node.data.outputShape = outputShape;
        } else {
          node.data.inputShape = cleanInputShapes[0];
          node.data.outputShape = outputShape;
        }
      }

      isUpdatingShapes.current = false;

      if (JSON.stringify(currentNodes) !== JSON.stringify(newNodes)) {
        return newNodes;
      }
      
      return currentNodes;
    });
  }, [edges, setNodes]);

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

  const isValidConnection = useCallback(
    (edgeOrConnection: Edge | Connection) => {
      const connection = edgeOrConnection as Connection;
      // For nodes with a specific target handle (e.g., multi-input nodes),
      // check if that specific handle is already connected.
      if (connection.targetHandle) {
        return !edges.some(
          (edge) => edge.target === connection.target && edge.targetHandle === connection.targetHandle,
        )
      }

      // For nodes without a specific target handle (e.g., single-input nodes),
      // check if the node is already a target for any edge.
      return !edges.some((edge) => edge.target === connection.target)
    },
    [edges],
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

      setNodes((nds) => {
        const existingIds = new Set(nds.map((n) => n.id));
        let maxNumericId = 0;
        nds.forEach((n) => {
          const match = n.id.match(/(?:[_-])(\d+)$/);
          if (match && match[1]) {
            const numericPart = parseInt(match[1], 10);
            if (numericPart > maxNumericId) {
              maxNumericId = numericPart;
            }
          }
        });

        let newNodeId = "";
        let counter = maxNumericId + 1;
        do {
          newNodeId = `${type}_${counter}`;
          counter++;
        } while (existingIds.has(newNodeId));
        
        const newNode: Node = {
            id: newNodeId,
            type,
            position,
            data,
        };
        return [...nds, newNode];
      });
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

  const updateNodeId = useCallback(
    (oldId: string, newId: string) => {
      takeSnapshot()
      setNodes((nds) => nds.map((n) => (n.id === oldId ? { ...n, id: newId } : n)))
      setEdges((eds) =>
        eds.map((e) => {
          const source = e.source === oldId ? newId : e.source
          const target = e.target === oldId ? newId : e.target
          return { ...e, source, target }
        }),
      )
      setSelectedNode((sn) => (sn && sn.id === oldId ? { ...sn, id: newId } : sn))
    },
    [setNodes, setEdges, setSelectedNode, takeSnapshot],
  )

  useEffect(() => {
    propagateTensorShapes()
  }, [nodes, edges, propagateTensorShapes])

  const loadExample = useCallback(
    async (exampleMetadata: ExampleNetworkMetadata) => {
      if (!exampleMetadata || typeof exampleMetadata !== 'object') {
        toast({
          title: "Loading Failed",
          description: "The selected example is invalid or empty.",
          variant: "destructive",
        });
        return;
      }

      try {
        const response = await fetch(`/api/examples/${exampleMetadata.filename}`);
        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`HTTP error! status: ${response.status}. ${errorText}`);
        }
        const example = await response.json();
        const { name, nodes, edges } = example;

        if (!name || !Array.isArray(nodes) || !Array.isArray(edges)) {
          toast({
            title: "Loading Failed",
            description: "Example file is malformed. It must contain a name, and arrays of nodes and edges.",
            variant: "destructive",
          });
          return;
        }

        // Check for nodes with missing critical properties
        for (const node of nodes) {
          if (!node.id || !node.type || !node.position || !node.data) {
            toast({
              title: "Loading Failed",
              description: `Example contains an invalid node object. Node ${node.id || '(ID missing)'} is missing required properties (id, type, position, data).`,
              variant: "destructive",
            });
            return;
          }
        }

        // Check for duplicate node IDs
        const nodeIds = new Set();
        for (const node of nodes) {
          if (nodeIds.has(node.id)) {
            toast({
              title: "Loading Failed",
              description: `Example contains duplicate node ID: ${node.id}.`,
              variant: "destructive",
            });
            return;
          }
          nodeIds.add(node.id);
        }

        const invalidNodeTypes = nodes.filter((node: any) => !nodeTypes[node.type]);
        if (invalidNodeTypes.length > 0) {
          toast({
            title: "Loading Failed",
            description: `Example contains unsupported node types: ${invalidNodeTypes.map((n: any) => n.type).join(", ")}`,
            variant: "destructive",
          });
          return;
        }

        // Check for edges with missing properties or pointing to non-existent nodes
        for (const edge of edges) {
          if (!edge.id || !edge.source || !edge.target) {
            toast({
              title: "Loading Failed",
              description: `Example contains an invalid edge object. Edge ${edge.id || '(ID missing)'} is missing required properties (id, source, target).`,
              variant: "destructive",
            });
            return;
          }
          if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
            toast({
              title: "Loading Failed",
              description: `Example contains an invalid edge: ${edge.id}. It connects to a non-existent node. Source: ${edge.source}, Target: ${edge.target}`,
              variant: "destructive",
            });
            return;
          }
        }

        takeSnapshot();
        setNodes(nodes);
        setEdges(edges);
        setSelectedNode(null);
        setCurrentModelName(name);

        toast({
          title: "Example Loaded",
          description: `Successfully loaded ${name}`,
        });

        setTimeout(() => {
          if (reactFlowInstanceRef.current) {
            reactFlowInstanceRef.current.fitView({ padding: 0.1, duration: 800 });
          }
        }, 100);
      } catch (error) {
        console.error("Failed to load example:", error);
        const fileName = exampleMetadata.filename;
        let description = `An unexpected error occurred while loading the example '${exampleMetadata.name}'.`;

        if (error instanceof Error) {
          if (error.message.toLowerCase().includes('failed to fetch')) {
            description = `Could not fetch the example file: '${fileName}'. Please check your network connection and ensure the file exists on the server at the expected location ('/api/examples/${fileName}').`;
          } else {
            description = `There was an issue processing the example file '${fileName}': ${error.message}`;
          }
        } else {
          description = `An unknown error occurred while trying to load '${fileName}'. See the console for more details.`;
        }

        toast({
          title: "Example Load Failed",
          description: description,
          variant: "destructive",
        });
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

    setTimeout(() => {
      if (reactFlowInstanceRef.current) {
        reactFlowInstanceRef.current.fitView({ padding: 0.1, duration: 200 })
      }
    }, 100)
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
    toast({ title: "Model Saved", description: `Model "${modelName}" has been saved.` })
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
          setSelectedNode(null);
          const importedModelName = importedData.name || file.name.replace('.json', '');
          setCurrentModelName(importedModelName);
          toast({ title: "Model Imported", description: `Successfully imported "${importedModelName}"` });
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
        setSelectedNode(null)
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

      const data = await response.json().catch(() => null)

      if (!response.ok || !data?.success) {
        throw new Error(data?.error || "Failed to generate model")
      }

      setGeneratedCode(data.code)

      setShowCodeDialog(true)
      toast({
        title: "Model Generated",
        description: "PyTorch model code generated successfully",
      })
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Failed to generate model code",
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

  // Shared clipboard helper with a secure (navigator.clipboard) path and an
  // execCommand fallback for insecure contexts. `onSuccess` runs after a
  // successful copy; failures surface a generic "Copy Failed" toast. All
  // window/navigator/document access lives inside this handler (SSR-safe).
  const copyTextToClipboard = useCallback(
    (text: string, onSuccess: () => void) => {
      const notifyFailure = () => {
        toast({
          title: "Copy Failed",
          description: "Failed to copy to clipboard",
          variant: "destructive",
        })
      }

      const unsecuredCopyToClipboard = (value: string) => {
        const textArea = document.createElement("textarea")
        textArea.value = value
        document.body.appendChild(textArea)
        textArea.focus()
        textArea.select()
        try {
          document.execCommand("copy")
          onSuccess()
        } catch {
          notifyFailure()
        }
        document.body.removeChild(textArea)
      }

      if (window.isSecureContext && navigator.clipboard) {
        navigator.clipboard.writeText(text).then(onSuccess).catch(notifyFailure)
      } else {
        unsecuredCopyToClipboard(text)
      }
    },
    [toast],
  )

  const copyCode = useCallback(() => {
    if (!generatedCode) return
    copyTextToClipboard(generatedCode, () => {
      setCopySuccess(true)
      toast({
        title: "Code Copied",
        description: "Generated code copied to clipboard",
      })
      setTimeout(() => setCopySuccess(false), 2000)
    })
  }, [generatedCode, copyTextToClipboard, toast])

  // Build a shareable URL that packs the current graph into the URL hash, and
  // copy it to the clipboard using the shared helper. SSR-safe: `window` is
  // only read inside this click handler.
  const handleShare = useCallback(() => {
    const url = buildShareUrl(window.location.origin + window.location.pathname, { nodes, edges })
    copyTextToClipboard(url, () => {
      toast({
        title: "Share link copied",
        description:
          url.length > 2000
            ? "The link is long because of the model size; some apps may truncate it."
            : "Paste the link anywhere to share this model.",
      })
    })
  }, [nodes, edges, copyTextToClipboard, toast])

  // Copy the already-computed model analysis as a torchinfo-style markdown table.
  const copyModelSummary = useCallback(() => {
    if (!modelAnalysis) return
    const summary = formatModelSummary(modelAnalysis, { format: "markdown" })
    copyTextToClipboard(summary, () => {
      toast({
        title: "Summary copied",
        description: "Model summary copied to clipboard as markdown.",
      })
    })
  }, [modelAnalysis, copyTextToClipboard, toast])

  // While the Code dialog is open, keep the shown code in sync with the training
  // options: generate WITHOUT training by default, or WITH the training scaffold
  // when the toggle is on. Regenerated client-side via ModelGenerator so option
  // changes take effect immediately.
  useEffect(() => {
    if (!showCodeDialog) return
    try {
      const generator = new ModelGenerator({ nodes, edges } as any)
      const code = includeTraining
        ? generator.generateCode({ training: { optimizer: trainingOptimizer, loss: trainingLoss } })
        : generator.generateCode()
      setGeneratedCode(code)
    } catch (error) {
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Failed to generate model code",
        variant: "destructive",
      })
    }
  }, [showCodeDialog, includeTraining, trainingOptimizer, trainingLoss, nodes, edges, toast])

  // --- Clipboard (copy/paste/cut) for canvas nodes ---
  const clipboardRef = useRef<{ nodes: Node[]; edges: Edge[] } | null>(null)

  const copySelectedNodes = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected)
    if (selectedNodes.length === 0) return false
    const selectedIds = new Set(selectedNodes.map((n) => n.id))
    const internalEdges = edges.filter((e) => selectedIds.has(e.source) && selectedIds.has(e.target))
    clipboardRef.current = {
      nodes: JSON.parse(JSON.stringify(selectedNodes)),
      edges: JSON.parse(JSON.stringify(internalEdges)),
    }
    toast({ title: "Copied", description: `Copied ${selectedNodes.length} node(s)` })
    return true
  }, [nodes, edges, toast])

  const pasteNodes = useCallback(() => {
    const clipboard = clipboardRef.current
    if (!clipboard || clipboard.nodes.length === 0) return
    takeSnapshot()

    const idMap = new Map<string, string>()
    const suffix = Date.now().toString(36)
    const pastedNodes = clipboard.nodes.map((node, i) => {
      const newId = `${node.type}_${suffix}_${i}`
      idMap.set(node.id, newId)
      return {
        ...node,
        id: newId,
        selected: true,
        position: { x: node.position.x + 40, y: node.position.y + 40 },
        data: JSON.parse(JSON.stringify(node.data)),
      }
    })
    const pastedEdges = clipboard.edges.map((edge, i) => ({
      ...edge,
      id: `edge_${suffix}_${i}`,
      source: idMap.get(edge.source)!,
      target: idMap.get(edge.target)!,
      selected: false,
    }))

    setNodes((nds) => [...nds.map((n) => ({ ...n, selected: false })), ...pastedNodes])
    setEdges((eds) => [...eds, ...pastedEdges])
    toast({ title: "Pasted", description: `Pasted ${pastedNodes.length} node(s)` })
  }, [setNodes, setEdges, takeSnapshot, toast])

  const duplicateNode = useCallback(
    (nodeToCopy: Node) => {
      takeSnapshot()
      const newId = `${nodeToCopy.type}_${Date.now().toString(36)}`
      const newNode: Node = {
        ...nodeToCopy,
        id: newId,
        selected: false,
        position: { x: nodeToCopy.position.x + 40, y: nodeToCopy.position.y + 40 },
        data: JSON.parse(JSON.stringify(nodeToCopy.data)),
      }
      setNodes((nds) => [...nds, newNode])
      toast({ title: "Node Duplicated", description: `Created ${newId}` })
    },
    [setNodes, takeSnapshot, toast],
  )

  const cutSelectedNodes = useCallback(() => {
    if (!copySelectedNodes()) return
    takeSnapshot()
    const selectedIds = new Set(nodes.filter((n) => n.selected).map((n) => n.id))
    setNodes((nds) => nds.filter((n) => !selectedIds.has(n.id)))
    setEdges((eds) => eds.filter((e) => !selectedIds.has(e.source) && !selectedIds.has(e.target)))
    if (selectedNode && selectedIds.has(selectedNode.id)) {
      setSelectedNode(null)
    }
  }, [copySelectedNodes, nodes, selectedNode, setNodes, setEdges, takeSnapshot])

  // Activate global keyboard shortcuts (they dispatch the CustomEvents handled below)
  useKeyboardShortcuts()

  useEffect(() => {
    const handlers: Record<string, () => void> = {
      "save-model": () => setShowSaveDialog(true),
      "open-model": handleOpenLoadDialog,
      "new-model": resetCanvas,
      "reset-canvas": resetCanvas,
      undo,
      redo,
      "generate-code": generateModel,
      "toggle-help": () => setShowHelpDialog((prev) => !prev),
      "fit-view": () => reactFlowInstanceRef.current?.fitView({ padding: 0.1, duration: 200 }),
      "select-all": () => setNodes((nds) => nds.map((n) => ({ ...n, selected: true }))),
      "deselect-all": () => setNodes((nds) => nds.map((n) => ({ ...n, selected: false }))),
      "copy-selected": copySelectedNodes,
      "paste-nodes": pasteNodes,
      "cut-selected": cutSelectedNodes,
    }

    const entries = Object.entries(handlers)
    entries.forEach(([event, handler]) => window.addEventListener(event, handler))
    return () => {
      entries.forEach(([event, handler]) => window.removeEventListener(event, handler))
    }
  }, [
    handleOpenLoadDialog,
    resetCanvas,
    undo,
    redo,
    generateModel,
    setNodes,
    copySelectedNodes,
    pasteNodes,
    cutSelectedNodes,
  ])

  // Convert the parser's logical graph into React-Flow nodes/edges. The parser
  // (lib/pytorch-parser.ts) reconstructs the *complete* computation graph from
  // the model's forward() method — branches, residuals, concatenations,
  // functional ops and inlined submodules — rather than a naive linear chain.
  const handleCodeInput = useCallback(() => {
    const result = parsePyTorchModel(inputCode)

    if (result.errors.length > 0) {
      setParseErrors(result.errors)
      setParseWarnings([])
      setUnsupportedModules([])
      return
    }

    const graphNodes = result.nodes.filter((n) => nodeTypes[n.type])
    if (graphNodes.length === 0) {
      setParseErrors(["No valid layers found in the code"])
      setParseWarnings([])
      setUnsupportedModules([])
      return
    }

    const validIds = new Set(graphNodes.map((n) => n.id))
    const newNodes: Node[] = graphNodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: n.data as any,
      draggable: true,
      selectable: true,
      deletable: true,
    }))
    const newEdges: Edge[] = result.edges
      .filter((e) => validIds.has(e.source) && validIds.has(e.target))
      .map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        type: e.type || "default",
        ...(e.targetHandle ? { targetHandle: e.targetHandle } : {}),
        animated: false,
        deletable: true,
      }))

    takeSnapshot()
    setNodes(newNodes)
    setEdges(newEdges)
    setSelectedNode(null)
    setParseErrors([])
    setParseWarnings(result.warnings)
    setUnsupportedModules(result.unsupportedModules)
    setShowCodeInputDialog(false)
    setInputCode("")

    const layerCount = newNodes.filter((n) => n.type !== "inputNode" && n.type !== "outputNode").length
    toast({
      title: "PyTorch Code Imported",
      description:
        `Reconstructed the full graph: ${layerCount} layers / ${newEdges.length} connections. ` +
        (result.unsupportedModules.length > 0
          ? `${result.unsupportedModules.length} unsupported module type(s) kept as pass-through nodes.`
          : "All modules are supported!"),
      duration: 5000,
    })
  }, [inputCode, toast, setNodes, setEdges, takeSnapshot])

  const keyboardShortcuts = [
    { key: "Ctrl + S", description: "Save model" },
    { key: "Ctrl + O", description: "Open model" },
    { key: "Ctrl + N", description: "New model (reset canvas)" },
    { key: "Ctrl + G", description: "Generate PyTorch code" },
    { key: "Ctrl + R", description: "Reset canvas" },
    { key: "Ctrl + Z", description: "Undo" },
    { key: "Ctrl + Y / Ctrl + Shift + Z", description: "Redo" },
    { key: "Ctrl + A", description: "Select all nodes" },
    { key: "Ctrl + D", description: "Deselect all nodes" },
    { key: "Ctrl + C", description: "Copy selected nodes" },
    { key: "Ctrl + X", description: "Cut selected nodes" },
    { key: "Ctrl + V", description: "Paste nodes" },
    { key: "Ctrl + F", description: "Fit view to canvas" },
    { key: "H", description: "Toggle help dialog" },
    { key: "Delete / Backspace", description: "Delete selected nodes" },
  ]

  const getNodeColor = (node: Node) => {
    switch (node.type) {
      case 'inputNode':
      case 'constantNode':
      case 'conv1dNode':
      case 'conv2dNode':
      case 'conv3dNode':
      case 'reshapeNode':
        return '#22c55e'; // green-500
      case 'linearNode':
      case 'embeddingNode':
      case 'downsampleNode':
        return '#3b82f6'; // blue-500
      case 'maxpool2dNode':
      case 'avgpool2dNode':
      case 'adaptiveavgpool2dNode':
        return '#ef4444'; // red-500
      case 'reluNode':
      case 'sigmoidNode':
      case 'tanhNode':
      case 'softmaxNode':
      case 'leakyreluNode':
      case 'geluNode':
      case 'siluNode':
      case 'mishNode':
      case 'hardswishNode':
      case 'hardsigmoidNode':
        return '#eab308'; // yellow-500
      case 'batchnorm1dNode':
      case 'batchnorm2dNode':
      case 'layernormNode':
      case 'rmsnormNode':
      case 'groupnormNode':
      case 'instancenorm1dNode':
      case 'instancenorm2dNode':
      case 'instancenorm3dNode':
        return '#06b6d4'; // cyan-500
      case 'moeNode':
        return '#d946ef'; // fuchsia-500
      case 'concatenateNode':
        return '#6366f1'; // indigo-500
      case 'addNode':
      case 'depthwiseconv2dNode':
      case 'transposeNode':
        return '#f97316'; // orange-500
      case 'multiplyNode':
      case 'mbconvNode':
      case 'separableconv2dNode':
      case 'convtranspose1dNode':
      case 'convtranspose2dNode':
      case 'convtranspose3dNode':
      case 'invertedResidualBlockNode':
      case 'transformerencoderlayerNode':
      case 'transformerdecoderlayerNode':
      case 'parameterNode':
        return '#a855f7'; // purple-500
      case 'lstmNode':
      case 'gruNode':
      case 'rnnNode':
      case 'multiheadattentionNode':
      case 'upsampleNode':
        return '#ec4899'; // pink-500
      case 'dropoutNode':
      case 'flattenNode':
      case 'seBlockNode':
      case 'seBottleneckNode':
        return '#6b7280'; // gray-500
      default:
        return '#ccc';
    }
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex items-center justify-between p-4 border-b border-border bg-card">
        <div className="flex items-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-2xl font-bold text-foreground">{currentModelName || "Neural Network Designer"}</h1>
            <p className="text-sm text-muted-foreground">Build PyTorch models visually</p>
          </div>
          {totalParameters > 0 && (
            <Badge variant="secondary" className="ml-2" title={`${totalParameters.toLocaleString()} trainable parameters`}>
              {formatNumber(totalParameters)} params
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
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
          <Button variant="outline" size="sm" onClick={handleShare}>
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <Network className="h-4 w-4 mr-2" />
                Load Example
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              {EXAMPLE_NETWORKS_METADATA.map((example) => (
                <DropdownMenuItem key={example.name} onClick={() => loadExample(example)}>
                  <div className="flex flex-col">
                    <span className="font-medium">{example.name}</span>
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
          <Button variant="outline" size="sm" onClick={() => setShowCodeInputDialog(true)}>
            <FileUp className="h-4 w-4 mr-2" />
            Import Code
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
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("parameterNode", { shape: [1, 1, 768] })}
                  >
                    <div className="flex items-center gap-2">
                      <Box className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Parameter</span>
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
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("embeddingNode", { num_embeddings: 30000, embedding_dim: 512 })}
                  >
                    <div className="flex items-center gap-2">
                      <Hash className="h-4 w-4 text-blue-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Embedding</span>
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
                    onClick={() => addNode("conv1dNode", { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv1D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("conv2dNode", { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv2D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("conv3dNode", { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Conv3D</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() =>
                      addNode("depthwiseconv2dNode", { in_channels: 32, out_channels: 32, kernel_size: 3, stride: 1, groups: 32 })
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
                      addNode("separableconv2dNode", { in_channels: 32, out_channels: 64, kernel_size: 3, stride: 1 })
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
                      addNode("convtranspose1dNode", { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1 })
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
                      addNode("convtranspose2dNode", { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 2 })
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
                      addNode("convtranspose3dNode", { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1 })
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
                    onClick={() => addNode("rmsnormNode", { normalized_shape: [128] })}
                  >
                    <div className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4 text-cyan-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">RMSNorm</span>
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
                    onClick={() => addNode("reshapeNode", { targetShape: "[-1, 784]" })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-green-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Reshape</span>
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
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("upsampleNode", { scale_factor: 2 })}
                  >
                    <div className="flex items-center gap-2">
                      <ArrowUpRight className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Upsample</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("downsampleNode", { scale_factor: 2 })}
                  >
                    <div className="flex items-center gap-2">
                      <ArrowDownLeft className="h-4 w-4 text-blue-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Downsample</span>
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
                    onClick={() => addNode("addNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <Plus className="h-4 w-4 text-orange-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Add (Skip Connection)</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("multiplyNode", {})}
                  >
                    <div className="flex items-center gap-2">
                      <X className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Multiply</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("concatenateNode", { dim: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <GitBranch className="h-4 w-4 text-indigo-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">Concatenate</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("mbconvNode", { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1, expand_ratio: 1, se_ratio: 0.25 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">MBConv</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("invertedResidualBlockNode", { in_channels: 32, out_channels: 16, stride: 1, expand_ratio: 1 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">InvertedResidualBlock</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("seBlockNode", { in_channels: 32, reduction: 16 })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">SE Block</span>
                    </div>
                  </Card>
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("seBottleneckNode", { in_planes: 64, planes: 64, stride: 1, downsample: false })}
                  >
                    <div className="flex items-center gap-2">
                      <Layers className="h-4 w-4 text-purple-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">SE Bottleneck</span>
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
                  <Card
                    className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                    onClick={() => addNode("rnnNode", { input_size: 128, hidden_size: 64 })}
                  >
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-pink-500" />
                      <span className="text-sm font-medium text-sidebar-foreground">RNN</span>
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
          <ErrorBoundary>
          <ReactFlowProvider>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              isValidConnection={isValidConnection}
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
              <MiniMap nodeColor={getNodeColor} />
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} />
            </ReactFlow>
          </ReactFlowProvider>
          </ErrorBoundary>
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
                  <div className="space-y-1">
                    <label className="text-sm font-medium text-sidebar-foreground/70">Node ID</label>
                    <Input
                      value={editingNodeId}
                      onChange={(e) => setEditingNodeId(e.target.value)}
                      onBlur={() => {
                        if (!selectedNode) return
                        const newId = editingNodeId.trim()
                        if (newId && newId !== selectedNode.id) {
                          if (nodes.some((n) => n.id === newId)) {
                            toast({
                              title: "ID already exists",
                              description: "Please choose a unique ID.",
                              variant: "destructive",
                            })
                            setEditingNodeId(selectedNode.id)
                          } else {
                            updateNodeId(selectedNode.id, newId)
                          }
                        } else if (newId !== selectedNode.id) {
                          setEditingNodeId(selectedNode.id)
                        }
                      }}
                      onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                        if (e.key === "Enter") {
                          ;(e.target as HTMLInputElement).blur()
                        } else if (e.key === "Escape") {
                          if (selectedNode) {
                            setEditingNodeId(selectedNode.id)
                          }
                          ;(e.target as HTMLInputElement).blur()
                        }
                      }}
                      className="w-full bg-background text-foreground"
                    />
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => duplicateNode(selectedNode)}
                >
                  <Copy className="h-3.5 w-3.5 mr-1.5" />
                  Duplicate Node
                </Button>
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
                        Shape: {formatTensorShape(selectedNode.data.outputShape as TensorShape | undefined)}
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
                        Shape: {formatTensorShape(selectedNode.data.outputShape as TensorShape | undefined)}
                      </div>
                    </>
                  )}
                  {selectedNode.type === "parameterNode" && (
                    <>
                      <div>
                        <label className="text-sm font-medium text-sidebar-foreground/70">Shape</label>
                        <Input
                          key={selectedNode.id}
                          defaultValue={JSON.stringify(selectedNode.data.shape || [])}
                          onBlur={(e) => {
                            try {
                              const newShape = JSON.parse(e.target.value);
                              if (Array.isArray(newShape) && newShape.every(item => typeof item === 'number')) {
                                updateNodeData(selectedNode.id, { shape: newShape });
                                return;
                              }
                            } catch (err) {
                              // fall through to reset below
                            }
                            // Invalid input: reset the field to the last valid shape
                            e.target.value = JSON.stringify(selectedNode.data.shape || []);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") {
                              (e.target as HTMLInputElement).blur();
                            }
                          }}
                          placeholder="e.g., [1, 1, 768]"
                          className="w-full bg-background text-foreground"
                        />
                        <p className="text-xs text-sidebar-foreground/70 mt-1">
                          Enter the shape as a JSON array of numbers. e.g., [1, 1, 768]
                        </p>
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
                  {selectedNode.type === "embeddingNode" && (
                    <>
                      <EditableNumberInput
                        label="Num Embeddings"
                        value={selectedNode.data.num_embeddings as number | undefined}
                        defaultValue={30000}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_embeddings: value })}
                      />
                      <EditableNumberInput
                        label="Embedding Dim"
                        value={selectedNode.data.embedding_dim as number | undefined}
                        defaultValue={512}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { embedding_dim: value })}
                      />
                      <EditableNumberInput
                        label="Padding Idx"
                        value={selectedNode.data.padding_idx as number | undefined}
                        defaultValue={0}
                        min={0}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { padding_idx: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "conv1dNode" && (
                    <>
                      <EditableNumberInput
                        label="Input Channels"
                        value={selectedNode.data.in_channels as number | undefined}
                        defaultValue={1}
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
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
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
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
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
                  {selectedNode.type === "conv3dNode" && (
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
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
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
                  {selectedNode.type === "depthwiseconv2dNode" && (
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
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
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
                        label="Groups"
                        value={selectedNode.data.groups as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { groups: value })}
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
                  {selectedNode.type === "mbconvNode" && (
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
                        label="Expand Ratio"
                        value={selectedNode.data.expand_ratio as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { expand_ratio: value })}
                      />
                      <EditableNumberInput
                        label="SE Ratio"
                        value={selectedNode.data.se_ratio as number | undefined}
                        defaultValue={0.25}
                        min={0}
                        max={1}
                        step={0.01}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { se_ratio: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "invertedResidualBlockNode" && (
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
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <EditableNumberInput
                        label="Expand Ratio"
                        value={selectedNode.data.expand_ratio as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { expand_ratio: value })}
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
                  {selectedNode.type === "concatenateNode" && (
                    <>
                      <div>
                        <label className="text-sm font-medium text-sidebar-foreground">Dimension (dim)</label>
                        <select
                          value={Number(selectedNode.data.dim ?? 1)}
                          onChange={(e) =>
                            updateNodeData(selectedNode.id, { dim: parseInt(e.target.value, 10) })
                          }
                          className="w-full px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500 bg-background text-foreground"
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
                  {selectedNode.type === "rmsnormNode" && (
                    <>
                      <EditableNumberInput
                        label="Normalized Shape"
                        value={
                          Array.isArray(selectedNode.data.normalized_shape)
                            ? (selectedNode.data.normalized_shape[selectedNode.data.normalized_shape.length - 1] as number)
                            : (selectedNode.data.normalized_shape as number | undefined)
                        }
                        defaultValue={128}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { normalized_shape: [value] })}
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
                  {selectedNode.type === "geluNode" && (
                    <div className="space-y-2">
                      <label htmlFor="gelu-approximate" className="text-sm font-medium">Approximate</label>
                      <select
                        id="gelu-approximate"
                        className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                        value={(selectedNode.data.approximate as string | undefined) ?? "none"}
                        onChange={(e) =>
                          updateNodeData(selectedNode.id, {
                            approximate: e.target.value === "none" ? undefined : e.target.value,
                          })
                        }
                      >
                        <option value="none">none (exact)</option>
                        <option value="tanh">tanh (approximation)</option>
                      </select>
                    </div>
                  )}
                  {selectedNode.type === "moeNode" && (
                    <>
                      <EditableNumberInput
                        label="d_model"
                        value={selectedNode.data.d_model as number | undefined}
                        defaultValue={512}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { d_model: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Expert Hidden Dim (d_ff)"
                        value={selectedNode.data.d_ff as number | undefined}
                        defaultValue={2048}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { d_ff: value })}
                      />
                      <EditableNumberInput
                        label="Number of Experts"
                        value={selectedNode.data.num_experts as number | undefined}
                        defaultValue={8}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { num_experts: value })}
                      />
                      <EditableNumberInput
                        label="Top-K (experts / token)"
                        value={selectedNode.data.top_k as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { top_k: value })}
                      />
                      <div className="space-y-2">
                        <label htmlFor="moe-activation" className="text-sm font-medium">Expert Activation</label>
                        <select
                          id="moe-activation"
                          className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                          value={(selectedNode.data.activation as string | undefined) ?? "gelu"}
                          onChange={(e) => updateNodeData(selectedNode.id, { activation: e.target.value })}
                        >
                          <option value="gelu">GELU</option>
                          <option value="silu">SiLU (Swish)</option>
                          <option value="relu">ReLU</option>
                        </select>
                      </div>
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
                  {selectedNode.type === "rnnNode" && (
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
                  {selectedNode.type === "reshapeNode" && (
                    <>
                      <div>
                        <label className="text-sm font-medium text-sidebar-foreground/70">Target Shape</label>
                        <Input
                          value={String(selectedNode.data.targetShape || "")}
                          onChange={(e) => updateNodeData(selectedNode.id, { targetShape: e.target.value })}
                          placeholder="e.g., [-1, 784]"
                          className="w-full bg-background text-foreground"
                        />
                        <p className="text-xs text-sidebar-foreground/70 mt-1">
                          Enter the target shape as a comma-separated list of integers, enclosed in square brackets. Use -1 to infer a dimension. e.g., [-1, 784] or [1, 28, 28]
                        </p>
                      </div>
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
                   {selectedNode.type === "upsampleNode" && (
                    <>
                      <EditableNumberInput
                        label="Scale Factor"
                        value={selectedNode.data.scale_factor as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { scale_factor: value })}
                      />
                    </>
                  )}
                   {selectedNode.type === "downsampleNode" && (
                    <>
                      <EditableNumberInput
                        label="Scale Factor"
                        value={selectedNode.data.scale_factor as number | undefined}
                        defaultValue={2}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { scale_factor: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "seBlockNode" && (
                    <>
                      <EditableNumberInput
                        label="In-Channels"
                        value={selectedNode.data.in_channels as number | undefined}
                        defaultValue={32}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_channels: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Reduction"
                        value={selectedNode.data.reduction as number | undefined}
                        defaultValue={16}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { reduction: value })}
                      />
                    </>
                  )}
                  {selectedNode.type === "seBottleneckNode" && (
                    <>
                      <EditableNumberInput
                        label="In-Planes"
                        value={selectedNode.data.in_planes as number | undefined}
                        defaultValue={64}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { in_planes: value })}
                        disabled={isInputConnected}
                      />
                      <EditableNumberInput
                        label="Planes"
                        value={selectedNode.data.planes as number | undefined}
                        defaultValue={64}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { planes: value })}
                      />
                      <EditableNumberInput
                        label="Stride"
                        value={selectedNode.data.stride as number | undefined}
                        defaultValue={1}
                        min={1}
                        onUpdate={(value) => updateNodeData(selectedNode.id, { stride: value })}
                      />
                      <div>
                        <label className="text-sm font-medium text-sidebar-foreground/70">Downsample</label>
                        <select
                          value={selectedNode.data.downsample ? 'true' : 'false'}
                          onChange={(e) =>
                            updateNodeData(selectedNode.id, { downsample: e.target.value === 'true' })
                          }
                          className="w-full px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500 bg-background text-foreground"
                        >
                          <option value="true">Yes</option>
                          <option value="false">No</option>
                        </select>
                      </div>
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
        <DialogContent className="w-[65vw] h-[80vh] flex flex-col bg-background text-foreground">
          <DialogHeader>
            <DialogTitle>Generated PyTorch Code</DialogTitle>
            <DialogDescription>
              Here is the PyTorch code for your model. You can copy it or download it as a Python file.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-wrap items-center gap-4 rounded-md border p-3 text-sm">
            <label className="flex cursor-pointer items-center gap-2">
              <input
                type="checkbox"
                checked={includeTraining}
                onChange={(e) => setIncludeTraining(e.target.checked)}
                className="h-4 w-4 accent-primary"
              />
              <span className="font-medium">Include training loop</span>
            </label>
            {includeTraining && (
              <>
                <div className="flex items-center gap-2">
                  <Label htmlFor="training-optimizer">Optimizer</Label>
                  <select
                    id="training-optimizer"
                    value={trainingOptimizer}
                    onChange={(e) => setTrainingOptimizer(e.target.value as TrainingOptions["optimizer"])}
                    className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                  >
                    <option value="adam">Adam</option>
                    <option value="adamw">AdamW</option>
                    <option value="sgd">SGD</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <Label htmlFor="training-loss">Loss</Label>
                  <select
                    id="training-loss"
                    value={trainingLoss}
                    onChange={(e) => setTrainingLoss(e.target.value as TrainingOptions["loss"])}
                    className="h-8 rounded-md border border-input bg-background px-2 text-sm"
                  >
                    <option value="crossentropy">CrossEntropy</option>
                    <option value="mse">MSE</option>
                    <option value="bce">BCE</option>
                  </select>
                </div>
              </>
            )}
          </div>
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
        <DialogContent className="max-w-4xl max-h-[90vh] w-[65vw] bg-background text-foreground">
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
                          className="flex items-center justify-between p-3 bg-muted rounded-lg min-w-0"
                        >
                          <div className="flex-1 min-w-0 mr-4">
                            <div className="font-medium text-sm truncate">{layer.name}</div>
                            <div className="text-xs text-muted-foreground truncate">{layer.type}</div>
                          </div>
                          <div className="flex gap-4 text-xs flex-shrink-0">
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-foreground">
                                {formatNumber(layer.parameters)}
                              </div>
                              <div className="text-muted-foreground">Params</div>
                            </div>
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-foreground">{formatNumber(layer.flops)}</div>
                              <div className="text-muted-foreground">FLOPs</div>
                            </div>
                            <div className="text-center min-w-0">
                              <div className="font-medium break-words text-foreground">
                                {layer.memoryMB.toFixed(1)} MB
                              </div>
                              <div className="text-muted-foreground">Memory</div>
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
          <div className="flex justify-end gap-2 pt-4 border-t">
            <Button variant="secondary" onClick={copyModelSummary} disabled={!modelAnalysis}>
              <Copy className="h-3.5 w-3.5 mr-1.5" />
              Copy summary
            </Button>
            <Button variant="default" onClick={() => setShowAnalysisPanel(false)}>
              Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog open={showValidationPanel} onOpenChange={setShowValidationPanel}>
        <DialogContent className="w-[65vw] h-[80vh] flex flex-col bg-background text-foreground">
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
                  <p className="text-sm text-muted-foreground">Your model is looking good!</p>
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
        <DialogContent className="w-[50vw] bg-background text-foreground">
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
              className="bg-background text-foreground"
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
        <DialogContent className="w-[50vw] bg-background text-foreground">
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
                      className="flex items-center justify-between rounded-lg border p-3 transition-all hover:bg-muted"
                    >
                      <div>
                        <div className="font-semibold">{model.name}</div>
                        <div className="text-xs text-muted-foreground">Saved: {new Date(model.timestamp).toLocaleString()}</div>
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
                  <div className="text-center text-muted-foreground py-10">No saved models found.</div>
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
        <DialogContent className="w-[80vw] h-[80vh] flex flex-col bg-background text-foreground">
          <DialogHeader>
            <DialogTitle>Input PyTorch Code</DialogTitle>
            <DialogDescription>
              Paste your PyTorch model code below. We parse the <code>forward()</code> method to
              reconstruct the complete computation graph — including residual/skip connections,
              concatenations, branches, functional ops and inlined submodules — not just a linear
              list of layers.
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
        <DialogContent className="max-w-4xl max-h-[90vh] w-[65vw] bg-background text-foreground">
          <DialogHeader>
            <DialogTitle>Neural Network Designer - Help Guide</DialogTitle>
          </DialogHeader>
          <ScrollArea className="h-[70vh] pr-4">
            <div className="space-y-6">
              {/* Getting Started */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">🚀 Getting Started</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>
                    Welcome to the Neural Network Designer! This tool helps you build PyTorch models visually by
                    connecting blocks on a canvas.
                  </p>
                </div>
              </div>

              {/* Adding Blocks */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📦 Adding Blocks</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>
                    <strong>1.</strong> Click on a block to select it (shows selection border)
                  </p>
                  <p>
                    <strong>2.</strong> Press the <kbd className="px-2 py-1 bg-muted rounded text-xs">Delete</kbd> or{' '}
                    <kbd className="px-2 py-1 bg-muted rounded text-xs">Backspace</kbd> key
                  </p>
                  <p>
                    <strong>3.</strong> The block and its connections will be removed
                  </p>
                </div>
              </div>

              {/* Keyboard Shortcuts */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">⌨️ Keyboard Shortcuts</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
                  {keyboardShortcuts.map((shortcut, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <p>{shortcut.description}</p>
                      <kbd className="px-2 py-1 bg-muted rounded text-xs font-semibold">{shortcut.key}</kbd>
                    </div>
                  ))}
                </div>
              </div>

              {/* Saving, Loading, Importing & Exporting */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">💾 Saving, Loading, Importing & Exporting</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>
                    <strong>To Save:</strong> Click the "Save" button, enter a name, and save it to your browser's local storage.
                  </p>
                  <p>
                    <strong>To Load:</strong> Click the "Open" button, and select a previously saved model from the list.
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="space-y-2 text-sm text-muted-foreground">
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
                <div className="grid grid-cols-2 gap-4 text-sm text-muted-foreground">
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

              {/* Feedback & Contact */}
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">📧 Feedback & Contact</h3>
                <div className="space-y-2 text-sm text-muted-foreground">
                  <p>
                    Have suggestions, found a bug, or want to request new features? Send your feedback to:{' '}
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
