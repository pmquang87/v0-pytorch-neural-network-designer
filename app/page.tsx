"use client"

import type React from "react"
import { useState, useCallback, useRef, useEffect } from "react"
import { type NodeTypes, ReactFlowProvider } from "reactflow"
import "reactflow/dist/style.css"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AlertCircle, Download, Upload, Play, HelpCircle, Save, Trash2, Copy } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

import { validateModel } from "@/lib/model-validator"
import { generatePyTorchCode } from "@/lib/code-generator"
import { autoSave, loadFromAutoSave } from "@/lib/auto-save"
import type { LayerType, LayerConfig, ValidationResult, NetworkState } from "@/lib/types"

interface CustomNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: {
    layerType: LayerType
    config: LayerConfig
    label: string
  }
}

interface CustomEdge {
  id: string
  source: string
  target: string
}

function LayerNodeComponent({
  node,
  isSelected,
  onSelect,
  onDragStart,
  onDelete,
}: {
  node: CustomNode
  isSelected: boolean
  onSelect: () => void
  onDragStart: (e: React.DragEvent) => void
  onDelete: () => void
}) {
  return (
    <div
      className={`absolute bg-white border-2 rounded-lg p-3 cursor-move min-w-[120px] shadow-md ${
        isSelected ? "border-blue-500 shadow-lg" : "border-gray-300"
      }`}
      style={{
        left: node.position.x,
        top: node.position.y,
        transform: "translate(-50%, -50%)",
      }}
      onClick={onSelect}
      draggable
      onDragStart={onDragStart}
    >
      <div className="text-sm font-medium text-center">{node.data.label}</div>
      <div className="text-xs text-gray-500 text-center mt-1">{node.data.layerType}</div>

      {/* Connection points */}
      <div
        className="absolute -top-2 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white connection-point"
        data-type="input"
      />
      <div
        className="absolute -bottom-2 left-1/2 transform -translate-x-1/2 w-4 h-4 bg-blue-500 rounded-full border-2 border-white connection-point"
        data-type="output"
      />

      {isSelected && (
        <button
          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs flex items-center justify-center"
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
        >
          ×
        </button>
      )}
    </div>
  )
}

function CustomCanvas({
  nodes,
  edges,
  selectedNode,
  onNodeSelect,
  onNodeMove,
  onNodeDelete,
  onConnect,
}: {
  nodes: CustomNode[]
  edges: CustomEdge[]
  selectedNode: CustomNode | null
  onNodeSelect: (node: CustomNode | null) => void
  onNodeMove: (nodeId: string, position: { x: number; y: number }) => void
  onNodeDelete: (nodeId: string) => void
  onConnect: (sourceId: string, targetId: string) => void
}) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const [draggedNode, setDraggedNode] = useState<string | null>(null)
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
  const [connecting, setConnecting] = useState<{ nodeId: string; type: "input" | "output" } | null>(null)

  const handleNodeDragStart = (e: React.DragEvent, node: CustomNode) => {
    setDraggedNode(node.id)
    const rect = canvasRef.current?.getBoundingClientRect()
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left - node.position.x,
        y: e.clientY - rect.top - node.position.y,
      })
    }
  }

  const handleCanvasDragOver = (e: React.DragEvent) => {
    e.preventDefault()
  }

  const handleCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (draggedNode) {
      const rect = canvasRef.current?.getBoundingClientRect()
      if (rect) {
        const x = e.clientX - rect.left - dragOffset.x
        const y = e.clientY - rect.top - dragOffset.y
        onNodeMove(draggedNode, { x, y })
      }
      setDraggedNode(null)
    }
  }

  const handleCanvasClick = (e: React.MouseEvent) => {
    if (e.target === canvasRef.current) {
      onNodeSelect(null)
    }
  }

  const renderConnections = () => {
    return edges.map((edge) => {
      const sourceNode = nodes.find((n) => n.id === edge.source)
      const targetNode = nodes.find((n) => n.id === edge.target)

      if (!sourceNode || !targetNode) return null

      return (
        <line
          key={edge.id}
          x1={sourceNode.position.x}
          y1={sourceNode.position.y + 20}
          x2={targetNode.position.x}
          y2={targetNode.position.y - 20}
          stroke="#3b82f6"
          strokeWidth="2"
          markerEnd="url(#arrowhead)"
        />
      )
    })
  }

  return (
    <div
      ref={canvasRef}
      className="relative w-full h-full bg-gray-50 overflow-hidden"
      onDragOver={handleCanvasDragOver}
      onDrop={handleCanvasDrop}
      onClick={handleCanvasClick}
    >
      {/* SVG for connections */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <defs>
          <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
          </marker>
        </defs>
        {renderConnections()}
      </svg>

      {/* Nodes */}
      {nodes.map((node) => (
        <LayerNodeComponent
          key={node.id}
          node={node}
          isSelected={selectedNode?.id === node.id}
          onSelect={() => onNodeSelect(node)}
          onDragStart={(e) => handleNodeDragStart(e, node)}
          onDelete={() => onNodeDelete(node.id)}
        />
      ))}

      {/* Grid background */}
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        <div
          className="w-full h-full"
          style={{
            backgroundImage: `
            linear-gradient(to right, #e5e7eb 1px, transparent 1px),
            linear-gradient(to bottom, #e5e7eb 1px, transparent 1px)
          `,
            backgroundSize: "20px 20px",
          }}
        />
      </div>
    </div>
  )
}

const nodeTypes: NodeTypes = {
  layer: LayerNodeComponent, // Use the custom component
}

const initialNodes: CustomNode[] = []
const initialEdges: CustomEdge[] = []

function NetworkDesigner() {
  const [nodes, setNodes] = useState<CustomNode[]>(initialNodes)
  const [edges, setEdges] = useState<CustomEdge[]>(initialEdges)
  const [selectedNode, setSelectedNode] = useState<CustomNode | null>(null)
  const [generatedCode, setGeneratedCode] = useState("")
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [showCodeDialog, setShowCodeDialog] = useState(false)
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false)
  const [showHelpDialog, setShowHelpDialog] = useState(false)
  const [showImportDialog, setShowImportDialog] = useState(false)
  const [importCode, setImportCode] = useState("")
  const [networkName, setNetworkName] = useState("MyNetwork")
  const [savedNetworks, setSavedNetworks] = useState<NetworkState[]>([])

  // Auto-save functionality
  useEffect(() => {
    const saveData = { nodes, edges, timestamp: Date.now() }
    autoSave(saveData)
  }, [nodes, edges])

  // Load saved networks on mount
  useEffect(() => {
    const saved = loadFromAutoSave()
    if (saved) {
      setNodes(saved.nodes)
      setEdges(saved.edges)
    }
  }, [])

  const handleNodeMove = useCallback((nodeId: string, position: { x: number; y: number }) => {
    setNodes((prev) => prev.map((node) => (node.id === nodeId ? { ...node, position } : node)))
  }, [])

  const handleConnect = useCallback((sourceId: string, targetId: string) => {
    const newEdge: CustomEdge = {
      id: `${sourceId}-${targetId}`,
      source: sourceId,
      target: targetId,
    }
    setEdges((prev) => [...prev, newEdge])
  }, [])

  const addLayer = useCallback((layerType: LayerType) => {
    const newNode: CustomNode = {
      id: `${layerType}-${Date.now()}`,
      type: "layer",
      position: {
        x: Math.random() * 400 + 200,
        y: Math.random() * 400 + 200,
      },
      data: {
        layerType,
        config: getDefaultConfig(layerType),
        label: layerType.charAt(0).toUpperCase() + layerType.slice(1),
      },
    }
    setNodes((prev) => [...prev, newNode])
  }, [])

  const updateNodeConfig = useCallback((nodeId: string, config: LayerConfig) => {
    setNodes((prev) => prev.map((node) => (node.id === nodeId ? { ...node, data: { ...node.data, config } } : node)))
  }, [])

  const deleteNode = useCallback(
    (nodeId: string) => {
      setNodes((prev) => prev.filter((node) => node.id !== nodeId))
      setEdges((prev) => prev.filter((edge) => edge.source !== nodeId && edge.target !== nodeId))
      if (selectedNode?.id === nodeId) {
        setSelectedNode(null)
      }
    },
    [selectedNode],
  )

  const duplicateNode = useCallback(
    (nodeId: string) => {
      const nodeToDuplicate = nodes.find((node) => node.id === nodeId)
      if (nodeToDuplicate) {
        const newNode: CustomNode = {
          ...nodeToDuplicate,
          id: `${nodeToDuplicate.data.layerType}-${Date.now()}`,
          position: {
            x: nodeToDuplicate.position.x + 50,
            y: nodeToDuplicate.position.y + 50,
          },
        }
        setNodes((prev) => [...prev, newNode])
      }
    },
    [nodes],
  )

  const validateNetwork = useCallback(async () => {
    setIsValidating(true)
    try {
      const reactFlowNodes = nodes.map((node) => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      }))
      const reactFlowEdges = edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      }))
      const result = await validateModel(reactFlowNodes, reactFlowEdges)
      setValidationResult(result)
    } catch (error) {
      console.error("Validation error:", error)
      setValidationResult({
        isValid: false,
        errors: ["Validation failed: " + (error as Error).message],
        warnings: [],
      })
    } finally {
      setIsValidating(false)
    }
  }, [nodes, edges])

  const generateCode = useCallback(async () => {
    setIsGenerating(true)
    try {
      const reactFlowNodes = nodes.map((node) => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      }))
      const reactFlowEdges = edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      }))
      const code = await generatePyTorchCode(reactFlowNodes, reactFlowEdges, networkName)
      setGeneratedCode(code)
      setShowCodeDialog(true)
    } catch (error) {
      console.error("Code generation error:", error)
      setGeneratedCode(`# Error generating code: ${(error as Error).message}`)
      setShowCodeDialog(true)
    } finally {
      setIsGenerating(false)
    }
  }, [nodes, edges, networkName])

  const clearNetwork = useCallback(() => {
    setNodes([])
    setEdges([])
    setSelectedNode(null)
    setValidationResult(null)
  }, [])

  const saveNetwork = useCallback(() => {
    const networkState: NetworkState = {
      nodes,
      edges,
      timestamp: Date.now(),
    }
    const saved = [...savedNetworks, networkState]
    setSavedNetworks(saved)
    localStorage.setItem("savedNetworks", JSON.stringify(saved))
  }, [nodes, edges, savedNetworks])

  const loadNetwork = useCallback((networkState: NetworkState) => {
    setNodes(networkState.nodes)
    setEdges(networkState.edges)
    setSelectedNode(null)
  }, [])

  function getDefaultConfig(layerType: LayerType): LayerConfig {
    switch (layerType) {
      case "linear":
        return { in_features: 128, out_features: 64, bias: true }
      case "conv2d":
        return { in_channels: 3, out_channels: 64, kernel_size: 3, stride: 1, padding: 1 }
      case "maxpool2d":
        return { kernel_size: 2, stride: 2, padding: 0 }
      case "avgpool2d":
        return { kernel_size: 2, stride: 2, padding: 0 }
      case "dropout":
        return { p: 0.5, inplace: false }
      case "batchnorm1d":
        return { num_features: 128, eps: 1e-5, momentum: 0.1 }
      case "batchnorm2d":
        return { num_features: 64, eps: 1e-5, momentum: 0.1 }
      case "relu":
        return { inplace: false }
      case "sigmoid":
        return {}
      case "tanh":
        return {}
      case "softmax":
        return { dim: 1 }
      case "leakyrelu":
        return { negative_slope: 0.01, inplace: false }
      case "elu":
        return { alpha: 1.0, inplace: false }
      case "gelu":
        return {}
      case "lstm":
        return {
          input_size: 128,
          hidden_size: 64,
          num_layers: 1,
          bias: true,
          batch_first: true,
          dropout: 0,
          bidirectional: false,
        }
      case "gru":
        return {
          input_size: 128,
          hidden_size: 64,
          num_layers: 1,
          bias: true,
          batch_first: true,
          dropout: 0,
          bidirectional: false,
        }
      case "embedding":
        return {
          num_embeddings: 1000,
          embedding_dim: 128,
          padding_idx: null,
          max_norm: null,
          norm_type: 2.0,
          scale_grad_by_freq: false,
          sparse: false,
        }
      case "layernorm":
        return { normalized_shape: [128], eps: 1e-5, elementwise_affine: true }
      case "multiheadattention":
        return {
          embed_dim: 128,
          num_heads: 8,
          dropout: 0.0,
          bias: true,
          add_bias_kv: false,
          add_zero_attn: false,
          kdim: null,
          vdim: null,
          batch_first: false,
        }
      case "transformer":
        return {
          d_model: 512,
          nhead: 8,
          num_encoder_layers: 6,
          num_decoder_layers: 6,
          dim_feedforward: 2048,
          dropout: 0.1,
          activation: "relu",
          custom_encoder: null,
          custom_decoder: null,
          layer_norm_eps: 1e-5,
          batch_first: false,
          norm_first: false,
        }
      default:
        return {}
    }
  }

  const renderPropertyPanel = () => {
    if (!selectedNode) {
      return (
        <Card>
          <CardHeader>
            <CardTitle>Properties</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Select a layer to edit its properties</p>
          </CardContent>
        </Card>
      )
    }

    const { layerType, config } = selectedNode.data

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            {layerType.charAt(0).toUpperCase() + layerType.slice(1)} Layer
            <div className="flex gap-1">
              <Button size="sm" variant="outline" onClick={() => duplicateNode(selectedNode.id)}>
                <Copy className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="outline" onClick={() => deleteNode(selectedNode.id)}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">{renderLayerProperties(layerType, config, selectedNode.id)}</CardContent>
      </Card>
    )
  }

  const renderLayerProperties = (layerType: LayerType, config: LayerConfig, nodeId: string) => {
    const updateConfig = (updates: Partial<LayerConfig>) => {
      updateNodeConfig(nodeId, { ...config, ...updates })
    }

    switch (layerType) {
      case "linear":
        return (
          <>
            <div>
              <Label htmlFor="in_features">Input Features</Label>
              <Input
                id="in_features"
                type="number"
                value={config.in_features || 128}
                onChange={(e) => updateConfig({ in_features: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="out_features">Output Features</Label>
              <Input
                id="out_features"
                type="number"
                value={config.out_features || 64}
                onChange={(e) => updateConfig({ out_features: Number.parseInt(e.target.value) })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="bias"
                checked={config.bias !== false}
                onChange={(e) => updateConfig({ bias: e.target.checked })}
              />
              <Label htmlFor="bias">Use Bias</Label>
            </div>
          </>
        )

      case "conv2d":
        return (
          <>
            <div>
              <Label htmlFor="in_channels">Input Channels</Label>
              <Input
                id="in_channels"
                type="number"
                value={config.in_channels || 3}
                onChange={(e) => updateConfig({ in_channels: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="out_channels">Output Channels</Label>
              <Input
                id="out_channels"
                type="number"
                value={config.out_channels || 64}
                onChange={(e) => updateConfig({ out_channels: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="kernel_size">Kernel Size</Label>
              <Input
                id="kernel_size"
                type="number"
                value={config.kernel_size || 3}
                onChange={(e) => updateConfig({ kernel_size: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="stride">Stride</Label>
              <Input
                id="stride"
                type="number"
                value={config.stride || 1}
                onChange={(e) => updateConfig({ stride: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="padding">Padding</Label>
              <Input
                id="padding"
                type="number"
                value={config.padding || 1}
                onChange={(e) => updateConfig({ padding: Number.parseInt(e.target.value) })}
              />
            </div>
          </>
        )

      case "maxpool2d":
      case "avgpool2d":
        return (
          <>
            <div>
              <Label htmlFor="kernel_size">Kernel Size</Label>
              <Input
                id="kernel_size"
                type="number"
                value={config.kernel_size || 2}
                onChange={(e) => updateConfig({ kernel_size: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="stride">Stride</Label>
              <Input
                id="stride"
                type="number"
                value={config.stride || 2}
                onChange={(e) => updateConfig({ stride: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="padding">Padding</Label>
              <Input
                id="padding"
                type="number"
                value={config.padding || 0}
                onChange={(e) => updateConfig({ padding: Number.parseInt(e.target.value) })}
              />
            </div>
          </>
        )

      case "dropout":
        return (
          <>
            <div>
              <Label htmlFor="p">Dropout Probability</Label>
              <Input
                id="p"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config.p || 0.5}
                onChange={(e) => updateConfig({ p: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="inplace"
                checked={config.inplace === true}
                onChange={(e) => updateConfig({ inplace: e.target.checked })}
              />
              <Label htmlFor="inplace">In-place</Label>
            </div>
          </>
        )

      case "batchnorm1d":
      case "batchnorm2d":
        return (
          <>
            <div>
              <Label htmlFor="num_features">Number of Features</Label>
              <Input
                id="num_features"
                type="number"
                value={config.num_features || (layerType === "batchnorm1d" ? 128 : 64)}
                onChange={(e) => updateConfig({ num_features: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="eps">Epsilon</Label>
              <Input
                id="eps"
                type="number"
                step="0.00001"
                value={config.eps || 1e-5}
                onChange={(e) => updateConfig({ eps: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="momentum">Momentum</Label>
              <Input
                id="momentum"
                type="number"
                step="0.01"
                value={config.momentum || 0.1}
                onChange={(e) => updateConfig({ momentum: Number.parseFloat(e.target.value) })}
              />
            </div>
          </>
        )

      case "relu":
      case "leakyrelu":
      case "elu":
        return (
          <>
            {layerType === "leakyrelu" && (
              <div>
                <Label htmlFor="negative_slope">Negative Slope</Label>
                <Input
                  id="negative_slope"
                  type="number"
                  step="0.01"
                  value={config.negative_slope || 0.01}
                  onChange={(e) => updateConfig({ negative_slope: Number.parseFloat(e.target.value) })}
                />
              </div>
            )}
            {layerType === "elu" && (
              <div>
                <Label htmlFor="alpha">Alpha</Label>
                <Input
                  id="alpha"
                  type="number"
                  step="0.1"
                  value={config.alpha || 1.0}
                  onChange={(e) => updateConfig({ alpha: Number.parseFloat(e.target.value) })}
                />
              </div>
            )}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="inplace"
                checked={config.inplace === true}
                onChange={(e) => updateConfig({ inplace: e.target.checked })}
              />
              <Label htmlFor="inplace">In-place</Label>
            </div>
          </>
        )

      case "softmax":
        return (
          <div>
            <Label htmlFor="dim">Dimension</Label>
            <Input
              id="dim"
              type="number"
              value={config.dim || 1}
              onChange={(e) => updateConfig({ dim: Number.parseInt(e.target.value) })}
            />
          </div>
        )

      case "lstm":
      case "gru":
        return (
          <>
            <div>
              <Label htmlFor="input_size">Input Size</Label>
              <Input
                id="input_size"
                type="number"
                value={config.input_size || 128}
                onChange={(e) => updateConfig({ input_size: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="hidden_size">Hidden Size</Label>
              <Input
                id="hidden_size"
                type="number"
                value={config.hidden_size || 64}
                onChange={(e) => updateConfig({ hidden_size: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="num_layers">Number of Layers</Label>
              <Input
                id="num_layers"
                type="number"
                value={config.num_layers || 1}
                onChange={(e) => updateConfig({ num_layers: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="dropout">Dropout</Label>
              <Input
                id="dropout"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config.dropout || 0}
                onChange={(e) => updateConfig({ dropout: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="bias"
                checked={config.bias !== false}
                onChange={(e) => updateConfig({ bias: e.target.checked })}
              />
              <Label htmlFor="bias">Use Bias</Label>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="batch_first"
                checked={config.batch_first === true}
                onChange={(e) => updateConfig({ batch_first: e.target.checked })}
              />
              <Label htmlFor="batch_first">Batch First</Label>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="bidirectional"
                checked={config.bidirectional === true}
                onChange={(e) => updateConfig({ bidirectional: e.target.checked })}
              />
              <Label htmlFor="bidirectional">Bidirectional</Label>
            </div>
          </>
        )

      case "embedding":
        return (
          <>
            <div>
              <Label htmlFor="num_embeddings">Number of Embeddings</Label>
              <Input
                id="num_embeddings"
                type="number"
                value={config.num_embeddings || 1000}
                onChange={(e) => updateConfig({ num_embeddings: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="embedding_dim">Embedding Dimension</Label>
              <Input
                id="embedding_dim"
                type="number"
                value={config.embedding_dim || 128}
                onChange={(e) => updateConfig({ embedding_dim: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="padding_idx">Padding Index</Label>
              <Input
                id="padding_idx"
                type="number"
                value={config.padding_idx || ""}
                onChange={(e) => updateConfig({ padding_idx: e.target.value ? Number.parseInt(e.target.value) : null })}
              />
            </div>
          </>
        )

      case "layernorm":
        return (
          <>
            <div>
              <Label htmlFor="normalized_shape">Normalized Shape</Label>
              <Input
                id="normalized_shape"
                value={Array.isArray(config.normalized_shape) ? config.normalized_shape.join(",") : "128"}
                onChange={(e) =>
                  updateConfig({ normalized_shape: e.target.value.split(",").map((x) => Number.parseInt(x.trim())) })
                }
              />
            </div>
            <div>
              <Label htmlFor="eps">Epsilon</Label>
              <Input
                id="eps"
                type="number"
                step="0.00001"
                value={config.eps || 1e-5}
                onChange={(e) => updateConfig({ eps: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="elementwise_affine"
                checked={config.elementwise_affine !== false}
                onChange={(e) => updateConfig({ elementwise_affine: e.target.checked })}
              />
              <Label htmlFor="elementwise_affine">Elementwise Affine</Label>
            </div>
          </>
        )

      case "multiheadattention":
        return (
          <>
            <div>
              <Label htmlFor="embed_dim">Embed Dimension</Label>
              <Input
                id="embed_dim"
                type="number"
                value={config.embed_dim || 128}
                onChange={(e) => updateConfig({ embed_dim: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="num_heads">Number of Heads</Label>
              <Input
                id="num_heads"
                type="number"
                value={config.num_heads || 8}
                onChange={(e) => updateConfig({ num_heads: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="dropout">Dropout</Label>
              <Input
                id="dropout"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config.dropout || 0.0}
                onChange={(e) => updateConfig({ dropout: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="bias"
                checked={config.bias !== false}
                onChange={(e) => updateConfig({ bias: e.target.checked })}
              />
              <Label htmlFor="bias">Use Bias</Label>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="batch_first"
                checked={config.batch_first === true}
                onChange={(e) => updateConfig({ batch_first: e.target.checked })}
              />
              <Label htmlFor="batch_first">Batch First</Label>
            </div>
          </>
        )

      case "transformer":
        return (
          <>
            <div>
              <Label htmlFor="d_model">Model Dimension</Label>
              <Input
                id="d_model"
                type="number"
                value={config.d_model || 512}
                onChange={(e) => updateConfig({ d_model: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="nhead">Number of Heads</Label>
              <Input
                id="nhead"
                type="number"
                value={config.nhead || 8}
                onChange={(e) => updateConfig({ nhead: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="num_encoder_layers">Encoder Layers</Label>
              <Input
                id="num_encoder_layers"
                type="number"
                value={config.num_encoder_layers || 6}
                onChange={(e) => updateConfig({ num_encoder_layers: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="num_decoder_layers">Decoder Layers</Label>
              <Input
                id="num_decoder_layers"
                type="number"
                value={config.num_decoder_layers || 6}
                onChange={(e) => updateConfig({ num_decoder_layers: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="dim_feedforward">Feedforward Dimension</Label>
              <Input
                id="dim_feedforward"
                type="number"
                value={config.dim_feedforward || 2048}
                onChange={(e) => updateConfig({ dim_feedforward: Number.parseInt(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="dropout">Dropout</Label>
              <Input
                id="dropout"
                type="number"
                step="0.1"
                min="0"
                max="1"
                value={config.dropout || 0.1}
                onChange={(e) => updateConfig({ dropout: Number.parseFloat(e.target.value) })}
              />
            </div>
            <div>
              <Label htmlFor="activation">Activation</Label>
              <Select
                value={config.activation || "relu"}
                onValueChange={(value) => updateConfig({ activation: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="relu">ReLU</SelectItem>
                  <SelectItem value="gelu">GELU</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id="batch_first"
                checked={config.batch_first === true}
                onChange={(e) => updateConfig({ batch_first: e.target.checked })}
              />
              <Label htmlFor="batch_first">Batch First</Label>
            </div>
          </>
        )

      default:
        return <p className="text-sm text-muted-foreground">No configurable properties</p>
    }
  }

  return (
    <div className="h-screen flex">
      {/* Sidebar */}
      <div className="w-80 bg-background border-r flex flex-col">
        {/* Header */}
        <div className="p-4 border-b">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold">PyTorch Designer</h1>
            <Dialog open={showHelpDialog} onOpenChange={setShowHelpDialog}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <HelpCircle className="h-4 w-4" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle>PyTorch Neural Network Designer - Help</DialogTitle>
                </DialogHeader>
                <div className="space-y-6">
                  <section>
                    <h3 className="text-lg font-semibold mb-2">Getting Started</h3>
                    <p className="text-sm text-muted-foreground mb-2">
                      Welcome to the PyTorch Neural Network Designer! This tool helps you visually design neural
                      networks and generate PyTorch code.
                    </p>
                    <ul className="text-sm space-y-1 ml-4">
                      <li>• Add layers from the sidebar by clicking on them</li>
                      <li>• Drag layers around the canvas to position them</li>
                      <li>• Select layers to edit their properties in the Properties panel</li>
                      <li>• Validate your network to check for errors</li>
                      <li>• Generate PyTorch code when your network is ready</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-lg font-semibold mb-2">Available Layers</h3>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <h4 className="font-medium mb-1">Core Layers</h4>
                        <ul className="space-y-1 ml-2">
                          <li>• Linear - Fully connected layer</li>
                          <li>• Conv2D - 2D convolution</li>
                          <li>• MaxPool2D - Max pooling</li>
                          <li>• AvgPool2D - Average pooling</li>
                          <li>• Dropout - Regularization</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium mb-1">Activation Functions</h4>
                        <ul className="space-y-1 ml-2">
                          <li>• ReLU - Rectified Linear Unit</li>
                          <li>• Sigmoid - Sigmoid activation</li>
                          <li>• Tanh - Hyperbolic tangent</li>
                          <li>• Softmax - Softmax activation</li>
                          <li>• LeakyReLU - Leaky ReLU</li>
                          <li>• ELU - Exponential Linear Unit</li>
                          <li>• GELU - Gaussian Error Linear Unit</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium mb-1">Normalization</h4>
                        <ul className="space-y-1 ml-2">
                          <li>• BatchNorm1D - 1D batch normalization</li>
                          <li>• BatchNorm2D - 2D batch normalization</li>
                          <li>• LayerNorm - Layer normalization</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium mb-1">Recurrent & Advanced</h4>
                        <ul className="space-y-1 ml-2">
                          <li>• LSTM - Long Short-Term Memory</li>
                          <li>• GRU - Gated Recurrent Unit</li>
                          <li>• Embedding - Embedding layer</li>
                          <li>• MultiheadAttention - Attention mechanism</li>
                          <li>• Transformer - Full transformer</li>
                        </ul>
                      </div>
                    </div>
                  </section>

                  <section>
                    <h3 className="text-lg font-semibold mb-2">Tips & Best Practices</h3>
                    <ul className="text-sm space-y-1 ml-4">
                      <li>• Always start with an input layer and end with an output layer</li>
                      <li>• Use batch normalization after convolutional layers for better training</li>
                      <li>• Add dropout layers to prevent overfitting</li>
                      <li>• Validate your network before generating code</li>
                      <li>• Save your work regularly using the auto-save feature</li>
                      <li>• Use appropriate activation functions for your task</li>
                    </ul>
                  </section>

                  <section>
                    <h3 className="text-lg font-semibold mb-2">Keyboard Shortcuts</h3>
                    <ul className="text-sm space-y-1 ml-4">
                      <li>• Delete - Remove selected layer</li>
                      <li>• Ctrl+C - Copy selected layer</li>
                      <li>• Ctrl+V - Paste layer</li>
                      <li>• Ctrl+Z - Undo (coming soon)</li>
                      <li>• Ctrl+S - Save network</li>
                    </ul>
                  </section>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="space-y-2">
            <div>
              <Label htmlFor="network-name">Network Name</Label>
              <Input
                id="network-name"
                value={networkName}
                onChange={(e) => setNetworkName(e.target.value)}
                placeholder="MyNetwork"
              />
            </div>

            <div className="flex gap-2">
              <Button onClick={validateNetwork} disabled={isValidating} size="sm" className="flex-1">
                {isValidating ? "Validating..." : "Validate"}
              </Button>
              <Button onClick={generateCode} disabled={isGenerating} size="sm" className="flex-1">
                <Play className="h-4 w-4 mr-1" />
                {isGenerating ? "Generating..." : "Generate"}
              </Button>
            </div>

            <div className="flex gap-2">
              <Button onClick={saveNetwork} variant="outline" size="sm">
                <Save className="h-4 w-4 mr-1" />
                Save
              </Button>
              <Button onClick={clearNetwork} variant="outline" size="sm">
                <Trash2 className="h-4 w-4 mr-1" />
                Clear
              </Button>
              <Dialog open={showImportDialog} onOpenChange={setShowImportDialog}>
                <DialogTrigger asChild>
                  <Button variant="outline" size="sm">
                    <Upload className="h-4 w-4 mr-1" />
                    Import
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Import PyTorch Code</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-4">
                    <Textarea
                      placeholder="Paste your PyTorch model code here..."
                      value={importCode}
                      onChange={(e) => setImportCode(e.target.value)}
                      rows={10}
                    />
                    <Button onClick={() => setShowImportDialog(false)}>Import Network</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>
          </div>
        </div>

        {/* Validation Results */}
        {validationResult && (
          <div className="p-4 border-b">
            <Alert variant={validationResult.isValid ? "default" : "destructive"}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {validationResult.isValid ? (
                  "Network is valid!"
                ) : (
                  <div>
                    <p className="font-medium">Validation Errors:</p>
                    <ul className="mt-1 text-sm">
                      {validationResult.errors.map((error, i) => (
                        <li key={i}>• {error}</li>
                      ))}
                    </ul>
                  </div>
                )}
                {validationResult.warnings.length > 0 && (
                  <div className="mt-2">
                    <p className="font-medium">Warnings:</p>
                    <ul className="mt-1 text-sm">
                      {validationResult.warnings.map((warning, i) => (
                        <li key={i}>• {warning}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Layer Palette */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-4">
            <h2 className="text-sm font-semibold mb-3">Layers</h2>
            <Tabs defaultValue="core" className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="core">Core</TabsTrigger>
                <TabsTrigger value="activation">Activation</TabsTrigger>
                <TabsTrigger value="norm">Norm</TabsTrigger>
                <TabsTrigger value="advanced">Advanced</TabsTrigger>
              </TabsList>

              <TabsContent value="core" className="space-y-2 mt-3">
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("linear")}
                >
                  Linear
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("conv2d")}
                >
                  Conv2D
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("maxpool2d")}
                >
                  MaxPool2D
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("avgpool2d")}
                >
                  AvgPool2D
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("dropout")}
                >
                  Dropout
                </Button>
              </TabsContent>

              <TabsContent value="activation" className="space-y-2 mt-3">
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("relu")}
                >
                  ReLU
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("sigmoid")}
                >
                  Sigmoid
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("tanh")}
                >
                  Tanh
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("softmax")}
                >
                  Softmax
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("leakyrelu")}
                >
                  LeakyReLU
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("elu")}
                >
                  ELU
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("gelu")}
                >
                  GELU
                </Button>
              </TabsContent>

              <TabsContent value="norm" className="space-y-2 mt-3">
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("batchnorm1d")}
                >
                  BatchNorm1D
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("batchnorm2d")}
                >
                  BatchNorm2D
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("layernorm")}
                >
                  LayerNorm
                </Button>
              </TabsContent>

              <TabsContent value="advanced" className="space-y-2 mt-3">
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("lstm")}
                >
                  LSTM
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("gru")}
                >
                  GRU
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("embedding")}
                >
                  Embedding
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("multiheadattention")}
                >
                  MultiheadAttention
                </Button>
                <Button
                  variant="outline"
                  className="w-full justify-start bg-transparent"
                  onClick={() => addLayer("transformer")}
                >
                  Transformer
                </Button>
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* Properties Panel */}
        <div className="border-t">
          <ScrollArea className="h-80">
            <div className="p-4">{renderPropertyPanel()}</div>
          </ScrollArea>
        </div>
      </div>

      {/* Main Canvas */}
      <div className="flex-1 relative">
        <CustomCanvas
          nodes={nodes}
          edges={edges}
          selectedNode={selectedNode}
          onNodeSelect={setSelectedNode}
          onNodeMove={handleNodeMove}
          onNodeDelete={deleteNode}
          onConnect={handleConnect}
        />
      </div>

      {/* Code Generation Dialog */}
      <Dialog open={showCodeDialog} onOpenChange={setShowCodeDialog}>
        <DialogContent className="max-w-4xl max-h-[80vh]">
          <DialogHeader>
            <DialogTitle>Generated PyTorch Code</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <Badge variant="secondary">{networkName}.py</Badge>
              <Button variant="outline" size="sm" onClick={() => navigator.clipboard.writeText(generatedCode)}>
                <Download className="h-4 w-4 mr-1" />
                Copy Code
              </Button>
            </div>
            <ScrollArea className="h-96">
              <pre className="text-sm bg-muted p-4 rounded-md overflow-x-auto">
                <code>{generatedCode}</code>
              </pre>
            </ScrollArea>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default function Home() {
  return (
    <ReactFlowProvider>
      <NetworkDesigner />
    </ReactFlowProvider>
  )
}
