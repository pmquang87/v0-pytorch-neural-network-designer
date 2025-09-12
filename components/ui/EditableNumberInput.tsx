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
}: {
  label: string
  value: number | undefined
  defaultValue: number
  min?: number
  step?: number
  max?: number
  onUpdate: (newValue: number) => void
}) => {
  const [inputValue, setInputValue] = useState(
    value !== undefined && value !== null ? value.toString() : defaultValue.toString()
  )
  const [isEditing, setIsEditing] = useState(false)

  useEffect(() => {
    setInputValue(value !== undefined && value !== null ? value.toString() : defaultValue.toString())
  }, [value, defaultValue])

  useEffect(() => {
    if (!isEditing && value !== undefined && value !== null) {
      setInputValue(value.toString())
    }
  }, [isEditing, value])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      const numValue = Number.parseInt(inputValue) || defaultValue
      onUpdate(numValue)
      setIsEditing(false)
      ;(e.currentTarget as HTMLInputElement).blur()
    }
  }

  const handleBlur = () => {
    const numValue = Number.parseInt(inputValue) || defaultValue
    onUpdate(numValue)
    setIsEditing(false)
  }

  const handleFocus = () => {
    setIsEditing(true)
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value)
  }

  return (
    <div>
      <label className="text-sm font-medium text-sidebar-foreground">{label}</label>
      <input
        type="number"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        onFocus={handleFocus}
        min={min}
        step={step}
        max={max}
        className="w-full px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
      />
    </div>
  )
}
