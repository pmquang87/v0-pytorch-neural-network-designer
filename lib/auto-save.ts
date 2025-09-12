import React from "react"
import { NetworkState } from "./types"

const STORAGE_KEY = "neural-network-designer-autosave"
const AUTO_SAVE_INTERVAL = 30000 // 30 seconds

export class AutoSaveManager {
  private intervalId: NodeJS.Timeout | null = null
  private lastSaveTime: number = 0
  private isEnabled: boolean = true

  constructor() {
    this.startAutoSave()
  }

  // Start auto-save timer
  startAutoSave(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
    }

    this.intervalId = setInterval(() => {
      this.autoSave()
    }, AUTO_SAVE_INTERVAL)
  }

  // Stop auto-save timer
  stopAutoSave(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  // Enable/disable auto-save
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled
    if (enabled) {
      this.startAutoSave()
    } else {
      this.stopAutoSave()
    }
  }

  // Manual save
  save(nodes: any[], edges: any[]): void {
    if (!this.isEnabled) return

    try {
      const state: NetworkState = {
        nodes: JSON.parse(JSON.stringify(nodes)), // Deep clone
        edges: JSON.parse(JSON.stringify(edges)), // Deep clone
        timestamp: Date.now()
      }

      localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
      this.lastSaveTime = Date.now()
      
      console.log("Model auto-saved to localStorage")
    } catch (error) {
      console.error("Failed to auto-save model:", error)
    }
  }

  // Load from localStorage
  load(): NetworkState | null {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      if (saved) {
        const state = JSON.parse(saved) as NetworkState
        console.log("Model loaded from localStorage")
        return state
      }
    } catch (error) {
      console.error("Failed to load model from localStorage:", error)
    }
    return null
  }

  // Clear saved data
  clear(): void {
    try {
      localStorage.removeItem(STORAGE_KEY)
      console.log("Auto-save data cleared")
    } catch (error) {
      console.error("Failed to clear auto-save data:", error)
    }
  }

  // Check if there's saved data
  hasSavedData(): boolean {
    return localStorage.getItem(STORAGE_KEY) !== null
  }

  // Get last save time
  getLastSaveTime(): number {
    return this.lastSaveTime
  }

  // Auto-save (called by timer)
  private autoSave(): void {
    if (!this.isEnabled) return

    // Dispatch event to get current state
    if (typeof window !== "undefined") {
      const event = new CustomEvent("auto-save-request")
      window.dispatchEvent(event)
    }
  }

  // Cleanup
  destroy(): void {
    this.stopAutoSave()
  }
}

// React hook for auto-save functionality
export function useAutoSave() {
  const [autoSaveManager] = React.useState(() => new AutoSaveManager())
  const [lastSaveTime, setLastSaveTime] = React.useState<number>(0)

  // Listen for auto-save requests
  React.useEffect(() => {
    if (typeof window === "undefined") return

    const handleAutoSaveRequest = (event: CustomEvent) => {
      // This will be handled by the main component
      // We need to create an event with nodes and edges data
      const saveEvent = new CustomEvent("auto-save-request-data")
      window.dispatchEvent(saveEvent)
    }

    window.addEventListener("auto-save-request", handleAutoSaveRequest as EventListener)

    return () => {
      window.removeEventListener("auto-save-request", handleAutoSaveRequest as EventListener)
    }
  }, [])

  // Listen for save triggers
  React.useEffect(() => {
    if (typeof window === "undefined") return

    const handleSaveTrigger = (event: CustomEvent) => {
      // Add null check before destructuring
      if (event.detail) {
        const { nodes, edges } = event.detail
        if (nodes && edges) {
          autoSaveManager.save(nodes, edges)
          setLastSaveTime(Date.now())
        }
      }
    }

    window.addEventListener("auto-save-trigger", handleSaveTrigger as EventListener)

    return () => {
      window.removeEventListener("auto-save-trigger", handleSaveTrigger as EventListener)
    }
  }, [autoSaveManager])

  React.useEffect(() => {
    return () => {
      autoSaveManager.destroy()
    }
  }, [autoSaveManager])

  return {
    save: autoSaveManager.save.bind(autoSaveManager),
    load: autoSaveManager.load.bind(autoSaveManager),
    clear: autoSaveManager.clear.bind(autoSaveManager),
    hasSavedData: autoSaveManager.hasSavedData.bind(autoSaveManager),
    setEnabled: autoSaveManager.setEnabled.bind(autoSaveManager),
    lastSaveTime
  }
}

// Utility functions for localStorage
export const StorageUtils = {
  // Save model with custom key
  saveModel: (key: string, nodes: any[], edges: any[]): void => {
    try {
      const state: NetworkState = {
        nodes: JSON.parse(JSON.stringify(nodes)),
        edges: JSON.parse(JSON.stringify(edges)),
        timestamp: Date.now()
      }
      localStorage.setItem(key, JSON.stringify(state))
    } catch (error) {
      console.error("Failed to save model:", error)
    }
  },

  // Load model with custom key
  loadModel: (key: string): NetworkState | null => {
    try {
      const saved = localStorage.getItem(key)
      if (saved) {
        return JSON.parse(saved) as NetworkState
      }
    } catch (error) {
      console.error("Failed to load model:", error)
    }
    return null
  },

  // Get all saved models
  getAllModels: (): Array<{ key: string; name: string; timestamp: number }> => {
    const models: Array<{ key: string; name: string; timestamp: number }> = []
    
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key && key.startsWith("nn-model-")) {
        try {
          const data = localStorage.getItem(key)
          if (data) {
            const state = JSON.parse(data) as NetworkState
            models.push({
              key,
              name: key.replace("nn-model-", ""),
              timestamp: state.timestamp
            })
          }
        } catch (error) {
          console.error("Failed to parse model:", key, error)
        }
      }
    }
    
    return models.sort((a, b) => b.timestamp - a.timestamp)
  },

  // Delete model
  deleteModel: (key: string): void => {
    localStorage.removeItem(key)
  },

  // Export model as JSON
  exportModel: (nodes: any[], edges: any[]): string => {
    const state: NetworkState = {
      nodes: JSON.parse(JSON.stringify(nodes)),
      edges: JSON.parse(JSON.stringify(edges)),
      timestamp: Date.now()
    }
    return JSON.stringify(state, null, 2)
  },

  // Import model from JSON
  importModel: (json: string): NetworkState | null => {
    try {
      return JSON.parse(json) as NetworkState
    } catch (error) {
      console.error("Failed to import model:", error)
      return null
    }
  }
}
