'use client';

import React from 'react';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
    BarChart3,
    Database,
    Box,
    Layers,
    Zap,
    Shrink,
    Eye,
    Plus,
    X,
    GitBranch,
    RotateCcw,
    Network,
    ArrowUpRight,
    ArrowDownLeft,
} from 'lucide-react';

interface LayerItem {
    type: string;
    label: string;
    icon: React.ReactNode;
    data: Record<string, any>;
}

interface LayerSection {
    title: string;
    items: LayerItem[];
}

// Define all available layers
const LAYER_SECTIONS: LayerSection[] = [
    {
        title: 'Input',
        items: [
            { type: 'inputNode', label: 'Input', icon: <Database className="h-4 w-4 text-purple-500" />, data: { channels: 3, height: 28, width: 28 } },
            { type: 'constantNode', label: 'Constant', icon: <Box className="h-4 w-4 text-green-500" />, data: { channels: 1, height: 1, width: 1 } },
            { type: 'parameterNode', label: 'Parameter', icon: <Box className="h-4 w-4 text-purple-500" />, data: { shape: [1, 1, 768] } },
        ],
    },
    {
        title: 'Linear',
        items: [
            { type: 'linearNode', label: 'Linear', icon: <BarChart3 className="h-4 w-4 text-blue-500" />, data: { in_features: 128, out_features: 64 } },
        ],
    },
    {
        title: 'Convolution',
        items: [
            { type: 'conv1dNode', label: 'Conv1D', icon: <Layers className="h-4 w-4 text-green-500" />, data: { in_channels: 1, out_channels: 32, kernel_size: 3, stride: 1 } },
            { type: 'conv2dNode', label: 'Conv2D', icon: <Layers className="h-4 w-4 text-green-500" />, data: { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 1 } },
            { type: 'conv3dNode', label: 'Conv3D', icon: <Layers className="h-4 w-4 text-green-500" />, data: { in_channels: 3, out_channels: 32, kernel_size: 3, stride: 1 } },
            { type: 'depthwiseconv2dNode', label: 'DepthwiseConv2D', icon: <Layers className="h-4 w-4 text-orange-500" />, data: { in_channels: 32, out_channels: 32, kernel_size: 3, stride: 1, groups: 32 } },
            { type: 'separableconv2dNode', label: 'SeparableConv2D', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 64, kernel_size: 3, stride: 1 } },
        ],
    },
    {
        title: 'Transposed Convolution',
        items: [
            { type: 'convtranspose1dNode', label: 'ConvTranspose1D', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1 } },
            { type: 'convtranspose2dNode', label: 'ConvTranspose2D', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 2 } },
            { type: 'convtranspose3dNode', label: 'ConvTranspose3D', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1 } },
        ],
    },
    {
        title: 'Activation',
        items: [
            { type: 'reluNode', label: 'ReLU', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'leakyreluNode', label: 'LeakyReLU', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: { negative_slope: 0.01 } },
            { type: 'geluNode', label: 'GELU', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'siluNode', label: 'SiLU', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'mishNode', label: 'Mish', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'hardswishNode', label: 'Hardswish', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'hardsigmoidNode', label: 'Hardsigmoid', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'sigmoidNode', label: 'Sigmoid', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'tanhNode', label: 'Tanh', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: {} },
            { type: 'softmaxNode', label: 'Softmax', icon: <Zap className="h-4 w-4 text-yellow-500" />, data: { dim: 1 } },
        ],
    },
    {
        title: 'Pooling',
        items: [
            { type: 'maxpool2dNode', label: 'MaxPool2D', icon: <Shrink className="h-4 w-4 text-red-500" />, data: { kernel_size: 2, stride: 2, padding: 0 } },
            { type: 'avgpool2dNode', label: 'AvgPool2D', icon: <Shrink className="h-4 w-4 text-red-500" />, data: { kernel_size: 2, stride: 2, padding: 0 } },
            { type: 'adaptiveavgpool2dNode', label: 'AdaptiveAvgPool2D', icon: <Shrink className="h-4 w-4 text-red-500" />, data: { output_size: [1, 1] } },
        ],
    },
    {
        title: 'Normalization',
        items: [
            { type: 'batchnorm1dNode', label: 'BatchNorm1D', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_features: 128 } },
            { type: 'batchnorm2dNode', label: 'BatchNorm2D', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_features: 32 } },
            { type: 'layernormNode', label: 'LayerNorm', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { normalized_shape: [128] } },
            { type: 'groupnormNode', label: 'GroupNorm', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_groups: 8, num_channels: 32 } },
            { type: 'instancenorm1dNode', label: 'InstanceNorm1D', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_features: 128 } },
            { type: 'instancenorm2dNode', label: 'InstanceNorm2D', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_features: 32 } },
            { type: 'instancenorm3dNode', label: 'InstanceNorm3D', icon: <BarChart3 className="h-4 w-4 text-cyan-500" />, data: { num_features: 16 } },
        ],
    },
    {
        title: 'Regularization',
        items: [
            { type: 'dropoutNode', label: 'Dropout', icon: <Eye className="h-4 w-4 text-gray-500" />, data: { p: 0.5 } },
        ],
    },
    {
        title: 'Utility',
        items: [
            { type: 'flattenNode', label: 'Flatten', icon: <Layers className="h-4 w-4 text-gray-500" />, data: {} },
            { type: 'reshapeNode', label: 'Reshape', icon: <Layers className="h-4 w-4 text-green-500" />, data: { targetShape: '[-1, 784]' } },
            { type: 'transposeNode', label: 'Transpose', icon: <RotateCcw className="h-4 w-4 text-orange-500" />, data: { dim0: 0, dim1: 1 } },
            { type: 'upsampleNode', label: 'Upsample', icon: <ArrowUpRight className="h-4 w-4 text-pink-500" />, data: { scale_factor: 2 } },
            { type: 'downsampleNode', label: 'Downsample', icon: <ArrowDownLeft className="h-4 w-4 text-blue-500" />, data: { scale_factor: 2 } },
        ],
    },
    {
        title: 'Advanced Operations',
        items: [
            { type: 'addNode', label: 'Add (Skip Connection)', icon: <Plus className="h-4 w-4 text-orange-500" />, data: {} },
            { type: 'multiplyNode', label: 'Multiply', icon: <X className="h-4 w-4 text-purple-500" />, data: {} },
            { type: 'concatenateNode', label: 'Concatenate', icon: <GitBranch className="h-4 w-4 text-indigo-500" />, data: { dim: 1 } },
            { type: 'mbconvNode', label: 'MBConv', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 16, kernel_size: 3, stride: 1, expand_ratio: 1, se_ratio: 0.25 } },
            { type: 'invertedResidualBlockNode', label: 'InvertedResidualBlock', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, out_channels: 16, stride: 1, expand_ratio: 1 } },
            { type: 'seBlockNode', label: 'SE Block', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_channels: 32, reduction: 16 } },
            { type: 'seBottleneckNode', label: 'SE Bottleneck', icon: <Layers className="h-4 w-4 text-purple-500" />, data: { in_planes: 64, planes: 64, stride: 1, downsample: false } },
        ],
    },
    {
        title: 'Recurrent',
        items: [
            { type: 'lstmNode', label: 'LSTM', icon: <Network className="h-4 w-4 text-pink-500" />, data: { input_size: 128, hidden_size: 64 } },
            { type: 'gruNode', label: 'GRU', icon: <Network className="h-4 w-4 text-pink-500" />, data: { input_size: 128, hidden_size: 64 } },
            { type: 'rnnNode', label: 'RNN', icon: <Network className="h-4 w-4 text-pink-500" />, data: { input_size: 128, hidden_size: 64 } },
        ],
    },
    {
        title: 'Attention',
        items: [
            { type: 'multiheadattentionNode', label: 'MultiheadAttention', icon: <Network className="h-4 w-4 text-pink-500" />, data: { embed_dim: 128, num_heads: 8 } },
            { type: 'scaledDotProductAttentionNode', label: 'ScaledDotProductAttention', icon: <Network className="h-4 w-4 text-pink-500" />, data: { embed_dim: 128 } },
            { type: 'transformerencoderlayerNode', label: 'TransformerEncoderLayer', icon: <Network className="h-4 w-4 text-pink-500" />, data: { d_model: 128, nhead: 8 } },
            { type: 'transformerdecoderlayerNode', label: 'TransformerDecoderLayer', icon: <Network className="h-4 w-4 text-pink-500" />, data: { d_model: 128, nhead: 8 } },
        ],
    },
    {
        title: 'State Space Models',
        items: [
            { type: 'ssmNode', label: 'SSM/Mamba', icon: <Network className="h-4 w-4 text-purple-500" />, data: { d_model: 128, d_state: 16, expand_factor: 2 } },
        ],
    },
    {
        title: 'Output',
        items: [
            { type: 'outputNode', label: 'Output', icon: <Database className="h-4 w-4 text-green-500" />, data: {} },
        ],
    },
];

interface LayerPaletteProps {
    onAddNode: (type: string, data: Record<string, any>) => void;
}

export function LayerPalette({ onAddNode }: LayerPaletteProps) {
    return (
        <div className="w-64 bg-sidebar border-r border-sidebar-border">
            <ScrollArea className="h-full">
                <div className="p-4 space-y-4">
                    <h2 className="font-semibold text-sidebar-foreground mb-3">Layer Library</h2>

                    {LAYER_SECTIONS.map((section) => (
                        <div key={section.title}>
                            <div className="text-sm text-sidebar-foreground/70 font-medium mb-2">
                                {section.title}
                            </div>
                            <div className="space-y-2">
                                {section.items.map((item) => (
                                    <Card
                                        key={item.type}
                                        className="p-3 cursor-pointer hover:bg-sidebar-accent/50 transition-colors border-sidebar-border"
                                        onClick={() => onAddNode(item.type, item.data)}
                                    >
                                        <div className="flex items-center gap-2">
                                            {item.icon}
                                            <span className="text-sm font-medium text-sidebar-foreground">
                                                {item.label}
                                            </span>
                                        </div>
                                    </Card>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </ScrollArea>
        </div>
    );
}

export { LAYER_SECTIONS };
