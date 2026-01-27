'use client';

import { Button } from '@/components/ui/button';
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
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
    Save,
    FolderOpen,
} from 'lucide-react';
import { EXAMPLE_NETWORKS_METADATA, type ExampleNetworkMetadata } from '@/lib/example-networks';

interface ToolbarProps {
    currentModelName: string | null;
    onHelp: () => void;
    onSave: () => void;
    onOpen: () => void;
    onLoadExample: (example: ExampleNetworkMetadata) => void;
    onUndo: () => void;
    onRedo: () => void;
    canUndo: boolean;
    canRedo: boolean;
    onReset: () => void;
    onAnalyze: () => void;
    onGenerate: () => void;
    isGenerating: boolean;
    nodesCount: number;
}

export function Toolbar({
    currentModelName,
    onHelp,
    onSave,
    onOpen,
    onLoadExample,
    onUndo,
    onRedo,
    canUndo,
    canRedo,
    onReset,
    onAnalyze,
    onGenerate,
    isGenerating,
    nodesCount,
}: ToolbarProps) {
    return (
        <div className="flex items-center justify-between p-4 border-b border-border bg-card">
            <div className="flex items-center gap-3">
                <Brain className="h-8 w-8 text-primary" />
                <div>
                    <h1 className="text-2xl font-bold text-foreground">
                        {currentModelName || 'Neural Network Designer'}
                    </h1>
                    <p className="text-sm text-muted-foreground">Build PyTorch models visually</p>
                </div>
            </div>
            <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={onHelp}>
                    <HelpCircle className="h-4 w-4 mr-2" />
                    Help
                </Button>
                <Button variant="outline" size="sm" onClick={onSave}>
                    <Save className="h-4 w-4 mr-2" />
                    Save
                </Button>
                <Button variant="outline" size="sm" onClick={onOpen}>
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
                    <DropdownMenuContent align="end" className="w-48 max-h-96 overflow-y-auto">
                        {EXAMPLE_NETWORKS_METADATA.map((example) => (
                            <DropdownMenuItem key={example.name} onClick={() => onLoadExample(example)}>
                                <div className="flex flex-col">
                                    <span className="font-medium">{example.name}</span>
                                </div>
                            </DropdownMenuItem>
                        ))}
                    </DropdownMenuContent>
                </DropdownMenu>
                <Button variant="outline" size="sm" onClick={onUndo} disabled={!canUndo}>
                    <Undo2 className="h-4 w-4 mr-2" />
                    Undo
                </Button>
                <Button variant="outline" size="sm" onClick={onRedo} disabled={!canRedo}>
                    <Redo2 className="h-4 w-4 mr-2" />
                    Redo
                </Button>
                <Button variant="outline" size="sm" onClick={onReset}>
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Reset
                </Button>
                <Button variant="outline" size="sm" onClick={onAnalyze} disabled={nodesCount === 0}>
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Model Analysis
                </Button>
                <Button onClick={onGenerate} disabled={isGenerating} className="flex items-center">
                    {isGenerating ? (
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                        <Code className="h-4 w-4 mr-2" />
                    )}
                    Generate PyTorch Code
                </Button>
            </div>
        </div>
    );
}
