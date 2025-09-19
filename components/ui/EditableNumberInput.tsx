"use client"

import type React from "react"
import { useState, useEffect } from "react"

export const EditableNumberInput = ({
  label,
  value,
  defaultValue,
  min = 0,
  step = 1,
  max,
  onUpdate,
  disabled = false,
}: {
  label: string
  value: number | undefined
  defaultValue: number
  min?: number
  step?: number
  max?: number
  onUpdate: (newValue: number) => void
  disabled?: boolean
}) => {
  const [inputValue, setInputValue] = useState(
    value !== undefined && value !== null ? value.toString() : defaultValue.toString()
  )
  const [isEditing, setIsEditing] = useState(false)
  const inputId = `editable-input-${label.replace(/\s+/g, "-").toLowerCase()}`

  useEffect(() => {
    if (!isEditing) {
      setInputValue(value !== undefined && value !== null ? value.toString() : defaultValue.toString())
    }
  }, [value, defaultValue, isEditing])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      const numValue = Number.parseFloat(inputValue) || defaultValue
      onUpdate(numValue)
      setIsEditing(false)
      ;(e.currentTarget as HTMLInputElement).blur()
    }
  }

  const handleBlur = () => {
    const numValue = Number.parseFloat(inputValue) || defaultValue
    onUpdate(numValue)
    setIsEditing(false)
  }

  const handleFocus = () => {
    if (disabled) return
    setIsEditing(true)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        className={`text-sm font-medium ${disabled ? "text-sidebar-foreground/50" : "text-sidebar-foreground"}`}
      >
        {label}
      </label>
      <input
        id={inputId}
        type="number"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        onFocus={handleFocus}
        min={min}
        step={step}
        max={max}
        disabled={disabled}
        className={`w-full px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500 ${
          disabled ? "bg-sidebar-accent/30 text-sidebar-foreground/50 cursor-not-allowed" : "bg-white text-black"
        }`}
      />
    </div>
  )
}
