import React from "react";
import { NetworkState } from "./types";

export class UndoRedoManager {
  private history: NetworkState[] = [];
  private currentIndex: number = -1;
  private maxHistorySize: number = 50;

  constructor(maxHistorySize: number = 50) {
    this.maxHistorySize = maxHistorySize;
  }

  // Save current state to history
  saveState(nodes: any[], edges: any[]): void {
    const newState: NetworkState = {
      nodes: JSON.parse(JSON.stringify(nodes)), // Deep clone
      edges: JSON.parse(JSON.stringify(edges)), // Deep clone
      timestamp: Date.now(),
    };

    // Remove any states after current index (when branching from history)
    this.history = this.history.slice(0, this.currentIndex + 1);

    // Add new state
    this.history.push(newState);
    this.currentIndex = this.history.length - 1;

    // Limit history size
    if (this.history.length > this.maxHistorySize) {
      this.history.shift();
      this.currentIndex--;
    }
  }

  // Undo to previous state
  undo(): NetworkState | null {
    if (this.canUndo()) {
      this.currentIndex--;
      return this.getCurrentState();
    }
    return null;
  }

  // Redo to next state
  redo(): NetworkState | null {
    if (this.canRedo()) {
      this.currentIndex++;
      return this.getCurrentState();
    }
    return null;
  }

  // Check if undo is possible
  canUndo(): boolean {
    return this.currentIndex > 0;
  }

  // Check if redo is possible
  canRedo(): boolean {
    return this.currentIndex < this.history.length - 1;
  }

  // Get current state
  getCurrentState(): NetworkState | null {
    if (this.currentIndex >= 0 && this.currentIndex < this.history.length) {
      return this.history[this.currentIndex];
    }
    return null;
  }

  // Clear all history
  clear(): void {
    this.history = [];
    this.currentIndex = -1;
  }

  // Get history info
  getHistoryInfo(): {
    current: number;
    total: number;
    canUndo: boolean;
    canRedo: boolean;
  } {
    return {
      current: this.currentIndex + 1,
      total: this.history.length,
      canUndo: this.canUndo(),
      canRedo: this.canRedo(),
    };
  }

  // Get all history states (for debugging)
  getAllStates(): NetworkState[] {
    return [...this.history];
  }
}

// React hook for undo/redo functionality
export function useUndoRedo(initialNodes: any[] = [], initialEdges: any[] = []) {
  const undoRedoManager = React.useRef(new UndoRedoManager());
  const [nodes, setNodes] = React.useState(initialNodes);
  const [edges, setEdges] = React.useState(initialEdges);
  const [historyInfo, setHistoryInfo] = React.useState(
    undoRedoManager.current.getHistoryInfo()
  );

  React.useEffect(() => {
    undoRedoManager.current.saveState(initialNodes, initialEdges);
    setHistoryInfo(undoRedoManager.current.getHistoryInfo());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const takeSnapshot = React.useCallback(() => {
    undoRedoManager.current.saveState(nodes, edges);
    setHistoryInfo(undoRedoManager.current.getHistoryInfo());
  }, [nodes, edges]);

  const undo = React.useCallback(() => {
    const state = undoRedoManager.current.undo();
    if (state) {
      setNodes(state.nodes);
      setEdges(state.edges);
      setHistoryInfo(undoRedoManager.current.getHistoryInfo());
    }
  }, []);

  const redo = React.useCallback(() => {
    const state = undoRedoManager.current.redo();
    if (state) {
      setNodes(state.nodes);
      setEdges(state.edges);
      setHistoryInfo(undoRedoManager.current.getHistoryInfo());
    }
  }, []);

  return {
    nodes,
    setNodes,
    edges,
    setEdges,
    undo,
    redo,
    takeSnapshot,
    canUndo: historyInfo.canUndo,
    canRedo: historyInfo.canRedo,
  };
}