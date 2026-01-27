import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { Node, Edge, NodeChange, EdgeChange, Connection } from '@xyflow/react';
import { applyNodeChanges, applyEdgeChanges, addEdge } from '@xyflow/react';
import type { TensorShape } from '../tensor-shape-calculator';

// Initial state
const initialNodes: Node[] = [
    {
        id: 'inputNode_1',
        type: 'inputNode',
        position: { x: 100, y: 100 },
        data: { channels: 3, height: 28, width: 28 },
    },
];

const initialEdges: Edge[] = [];

// Network state interface
interface NetworkState {
    // State
    nodes: Node[];
    edges: Edge[];
    selectedNode: Node | null;
    currentModelName: string | null;

    // History for undo/redo
    history: { nodes: Node[]; edges: Edge[] }[];
    historyIndex: number;

    // Actions
    setNodes: (nodes: Node[] | ((nodes: Node[]) => Node[])) => void;
    setEdges: (edges: Edge[] | ((edges: Edge[]) => Edge[])) => void;
    onNodesChange: (changes: NodeChange[]) => void;
    onEdgesChange: (changes: EdgeChange[]) => void;
    onConnect: (connection: Connection) => void;

    addNode: (type: string, data: any, position?: { x: number; y: number }) => void;
    updateNodeData: (nodeId: string, newData: any) => void;
    updateNodeId: (oldId: string, newId: string) => void;
    setSelectedNode: (node: Node | null) => void;
    setCurrentModelName: (name: string | null) => void;

    resetCanvas: () => void;
    loadNetwork: (nodes: Node[], edges: Edge[], modelName?: string) => void;

    // Undo/Redo
    takeSnapshot: () => void;
    undo: () => void;
    redo: () => void;
    canUndo: () => boolean;
    canRedo: () => boolean;
}

const MAX_HISTORY_SIZE = 50;

export const useNetworkStore = create<NetworkState>()((set, get) => ({
    // Initial state
    nodes: initialNodes,
    edges: initialEdges,
    selectedNode: null,
    currentModelName: null,
    history: [{ nodes: initialNodes, edges: initialEdges }],
    historyIndex: 0,

    // Setters
    setNodes: (nodesOrFn) => {
        set((state) => ({
            nodes: typeof nodesOrFn === 'function' ? nodesOrFn(state.nodes) : nodesOrFn,
        }));
    },

    setEdges: (edgesOrFn) => {
        set((state) => ({
            edges: typeof edgesOrFn === 'function' ? edgesOrFn(state.edges) : edgesOrFn,
        }));
    },

    onNodesChange: (changes) => {
        const removals = changes.filter((c) => c.type === 'remove');
        if (removals.length > 0) {
            get().takeSnapshot();
            const state = get();
            if (state.selectedNode && removals.some((r) => r.id === state.selectedNode?.id)) {
                set({ selectedNode: null });
            }
        }
        set((state) => ({
            nodes: applyNodeChanges(changes, state.nodes),
        }));
    },

    onEdgesChange: (changes) => {
        if (changes.some((c) => c.type === 'remove')) {
            get().takeSnapshot();
        }
        set((state) => ({
            edges: applyEdgeChanges(changes, state.edges),
        }));
    },

    onConnect: (connection) => {
        get().takeSnapshot();
        set((state) => ({
            edges: addEdge(connection, state.edges),
        }));
    },

    addNode: (type, data, position) => {
        get().takeSnapshot();
        set((state) => {
            // Generate unique ID
            const existingIds = new Set(state.nodes.map((n) => n.id));
            let maxNumericId = 0;
            state.nodes.forEach((n) => {
                const match = n.id.match(/(?:[_-])(\d+)$/);
                if (match && match[1]) {
                    const numericPart = parseInt(match[1], 10);
                    if (numericPart > maxNumericId) {
                        maxNumericId = numericPart;
                    }
                }
            });

            let newNodeId = '';
            let counter = maxNumericId + 1;
            do {
                newNodeId = `${type}_${counter}`;
                counter++;
            } while (existingIds.has(newNodeId));

            const newNode: Node = {
                id: newNodeId,
                type,
                position: position || { x: Math.random() * 400 + 100, y: Math.random() * 400 + 100 },
                data,
            };

            return { nodes: [...state.nodes, newNode] };
        });
    },

    updateNodeData: (nodeId, newData) => {
        get().takeSnapshot();
        set((state) => {
            const updatedNodes = state.nodes.map((node) => {
                if (node.id === nodeId) {
                    return { ...node, data: { ...node.data, ...newData } };
                }
                return node;
            });

            // Update selected node if it's the one being updated
            const selectedNode = state.selectedNode;
            let updatedSelectedNode = selectedNode;
            if (selectedNode && selectedNode.id === nodeId) {
                updatedSelectedNode = updatedNodes.find((n) => n.id === nodeId) || null;
            }

            return { nodes: updatedNodes, selectedNode: updatedSelectedNode };
        });
    },

    updateNodeId: (oldId, newId) => {
        get().takeSnapshot();
        set((state) => ({
            nodes: state.nodes.map((n) => (n.id === oldId ? { ...n, id: newId } : n)),
            edges: state.edges.map((e) => ({
                ...e,
                source: e.source === oldId ? newId : e.source,
                target: e.target === oldId ? newId : e.target,
            })),
            selectedNode:
                state.selectedNode && state.selectedNode.id === oldId
                    ? { ...state.selectedNode, id: newId }
                    : state.selectedNode,
        }));
    },

    setSelectedNode: (node) => set({ selectedNode: node }),

    setCurrentModelName: (name) => set({ currentModelName: name }),

    resetCanvas: () => {
        get().takeSnapshot();
        set({
            nodes: initialNodes,
            edges: initialEdges,
            selectedNode: null,
            currentModelName: null,
        });
    },

    loadNetwork: (nodes, edges, modelName) => {
        get().takeSnapshot();
        set({
            nodes,
            edges,
            selectedNode: null,
            currentModelName: modelName || null,
        });
    },

    // Undo/Redo implementation
    takeSnapshot: () => {
        set((state) => {
            const newHistory = state.history.slice(0, state.historyIndex + 1);
            newHistory.push({
                nodes: JSON.parse(JSON.stringify(state.nodes)),
                edges: JSON.parse(JSON.stringify(state.edges)),
            });

            // Limit history size
            if (newHistory.length > MAX_HISTORY_SIZE) {
                newHistory.shift();
            }

            return {
                history: newHistory,
                historyIndex: newHistory.length - 1,
            };
        });
    },

    undo: () => {
        const state = get();
        if (state.historyIndex > 0) {
            const newIndex = state.historyIndex - 1;
            const snapshot = state.history[newIndex];
            set({
                nodes: snapshot.nodes,
                edges: snapshot.edges,
                historyIndex: newIndex,
            });
        }
    },

    redo: () => {
        const state = get();
        if (state.historyIndex < state.history.length - 1) {
            const newIndex = state.historyIndex + 1;
            const snapshot = state.history[newIndex];
            set({
                nodes: snapshot.nodes,
                edges: snapshot.edges,
                historyIndex: newIndex,
            });
        }
    },

    canUndo: () => get().historyIndex > 0,
    canRedo: () => get().historyIndex < get().history.length - 1,
}));

// Selector hooks for common operations
export const useNodes = () => useNetworkStore((state) => state.nodes);
export const useEdges = () => useNetworkStore((state) => state.edges);
export const useSelectedNode = () => useNetworkStore((state) => state.selectedNode);
export const useCurrentModelName = () => useNetworkStore((state) => state.currentModelName);
