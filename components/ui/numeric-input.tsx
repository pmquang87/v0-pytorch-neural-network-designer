import * as React from "react"
import { cn } from "@/lib/utils"

export interface NumericInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange' | 'value'> {
  value: number
  onValueChange: (value: number) => void
  min?: number
  max?: number
}

const NumericInput = React.forwardRef<HTMLInputElement, NumericInputProps>(
  ({ className, value, onValueChange, min, max, ...props }, ref) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      const num = parseInt(e.target.value, 10)
      if (!isNaN(num)) {
        if ((min === undefined || num >= min) && (max === undefined || num <= max)) {
          onValueChange(num)
        }
      }
    }

    return (
      <input
        type="number"
        className={cn(
          "flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",
          className,
        )}
        ref={ref}
        value={value}
        onChange={handleChange}
        min={min}
        max={max}
        {...props}
      />
    )
  },
)
NumericInput.displayName = "NumericInput"

export { NumericInput }
