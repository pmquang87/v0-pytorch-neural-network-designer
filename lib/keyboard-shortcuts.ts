import React from "react"
import { KeyboardShortcut } from "./types"

export class KeyboardShortcutManager {
  private shortcuts: Map<string, KeyboardShortcut> = new Map()
  private isEnabled: boolean = true

  constructor() {
    this.setupGlobalListener()
  }

  // Register a keyboard shortcut
  register(shortcut: KeyboardShortcut): void {
    const key = this.getKeyString(shortcut)
    this.shortcuts.set(key, shortcut)
  }

  // Unregister a keyboard shortcut
  unregister(key: string, ctrlKey?: boolean, shiftKey?: boolean, altKey?: boolean): void {
    const keyString = this.getKeyString({ key, ctrlKey, shiftKey, altKey, action: () => {}, description: "" })
    this.shortcuts.delete(keyString)
  }

  // Enable/disable all shortcuts
  setEnabled(enabled: boolean): void {
    this.isEnabled = enabled
  }

  // Get all registered shortcuts
  getAllShortcuts(): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values())
  }

  // Setup global keyboard listener
  private setupGlobalListener(): void {
    if (typeof document !== "undefined") {
      document.addEventListener("keydown", this.handleKeyDown.bind(this))
    }
  }

  // Handle keydown events
  private handleKeyDown(event: KeyboardEvent): void {
    if (!this.isEnabled) return

    // Don't trigger shortcuts when typing in input fields
    const target = event.target as HTMLElement
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.contentEditable === "true") {
      return
    }

    const keyString = this.getKeyString({
      key: event.key,
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey,
      altKey: event.altKey,
      action: () => {},
      description: ""
    })

    const shortcut = this.shortcuts.get(keyString)
    if (shortcut) {
      event.preventDefault()
      shortcut.action()
    }
  }

  // Generate key string for shortcut identification
  private getKeyString(shortcut: Partial<KeyboardShortcut>): string {
    const parts: string[] = []
    if (shortcut.ctrlKey) parts.push("ctrl")
    if (shortcut.shiftKey) parts.push("shift")
    if (shortcut.altKey) parts.push("alt")
    parts.push(shortcut.key?.toLowerCase() || "")
    return parts.join("+")
  }

  // Cleanup
  destroy(): void {
    if (typeof document !== "undefined") {
      document.removeEventListener("keydown", this.handleKeyDown.bind(this))
    }
    this.shortcuts.clear()
  }
}

// Default shortcuts for the neural network designer
export const DEFAULT_SHORTCUTS: KeyboardShortcut[] = [
  {
    key: "s",
    ctrlKey: true,
    action: () => {
      // Save model
      const event = new CustomEvent("save-model")
      window.dispatchEvent(event)
    },
    description: "Save model"
  },
  {
    key: "o",
    ctrlKey: true,
    action: () => {
      // Open model
      const event = new CustomEvent("open-model")
      window.dispatchEvent(event)
    },
    description: "Open model"
  },
  {
    key: "n",
    ctrlKey: true,
    action: () => {
      // New model
      const event = new CustomEvent("new-model")
      window.dispatchEvent(event)
    },
    description: "New model"
  },
  {
    key: "z",
    ctrlKey: true,
    action: () => {
      // Undo
      const event = new CustomEvent("undo")
      window.dispatchEvent(event)
    },
    description: "Undo"
  },
  {
    key: "y",
    ctrlKey: true,
    action: () => {
      // Redo
      const event = new CustomEvent("redo")
      window.dispatchEvent(event)
    },
    description: "Redo"
  },
  {
    key: "z",
    ctrlKey: true,
    shiftKey: true,
    action: () => {
      // Redo (alternative)
      const event = new CustomEvent("redo")
      window.dispatchEvent(event)
    },
    description: "Redo (alternative)"
  },
  {
    key: "g",
    ctrlKey: true,
    action: () => {
      // Generate code
      const event = new CustomEvent("generate-code")
      window.dispatchEvent(event)
    },
    description: "Generate PyTorch code"
  },
  {
    key: "r",
    ctrlKey: true,
    action: () => {
      // Reset canvas
      const event = new CustomEvent("reset-canvas")
      window.dispatchEvent(event)
    },
    description: "Reset canvas"
  },
  {
    key: "a",
    ctrlKey: true,
    action: () => {
      // Select all nodes
      const event = new CustomEvent("select-all")
      window.dispatchEvent(event)
    },
    description: "Select all nodes"
  },
  {
    key: "d",
    ctrlKey: true,
    action: () => {
      // Deselect all nodes
      const event = new CustomEvent("deselect-all")
      window.dispatchEvent(event)
    },
    description: "Deselect all nodes"
  },
  {
    key: "c",
    ctrlKey: true,
    action: () => {
      // Copy selected nodes
      const event = new CustomEvent("copy-selected")
      window.dispatchEvent(event)
    },
    description: "Copy selected nodes"
  },
  {
    key: "v",
    ctrlKey: true,
    action: () => {
      // Paste nodes
      const event = new CustomEvent("paste-nodes")
      window.dispatchEvent(event)
    },
    description: "Paste nodes"
  },
  {
    key: "x",
    ctrlKey: true,
    action: () => {
      // Cut selected nodes
      const event = new CustomEvent("cut-selected")
      window.dispatchEvent(event)
    },
    description: "Cut selected nodes"
  },
  {
    key: "f",
    ctrlKey: true,
    action: () => {
      // Fit view
      const event = new CustomEvent("fit-view")
      window.dispatchEvent(event)
    },
    description: "Fit view to canvas"
  },
  {
    key: "h",
    action: () => {
      // Toggle help
      const event = new CustomEvent("toggle-help")
      window.dispatchEvent(event)
    },
    description: "Toggle help dialog"
  }
]

// React hook for keyboard shortcuts
export function useKeyboardShortcuts(shortcuts: KeyboardShortcut[] = []) {
  const [shortcutManager] = React.useState(() => new KeyboardShortcutManager())

  React.useEffect(() => {
    // Register default shortcuts
    DEFAULT_SHORTCUTS.forEach(shortcut => {
      shortcutManager.register(shortcut)
    })

    // Register custom shortcuts
    shortcuts.forEach(shortcut => {
      shortcutManager.register(shortcut)
    })

    return () => {
      shortcutManager.destroy()
    }
  }, [shortcutManager, shortcuts])

  return {
    register: shortcutManager.register.bind(shortcutManager),
    unregister: shortcutManager.unregister.bind(shortcutManager),
    setEnabled: shortcutManager.setEnabled.bind(shortcutManager),
    getAllShortcuts: shortcutManager.getAllShortcuts.bind(shortcutManager)
  }
}
