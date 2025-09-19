import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export const formatShape = (shape: any) => {
  if (Array.isArray(shape)) {
    return shape.join(' x ');
  }
  if (typeof shape === 'object' && shape !== null) {
    if ('channels' in shape && 'height' in shape && 'width' in shape) {
      return `${shape.channels} x ${shape.height} x ${shape.width}`;
    }
    return JSON.stringify(shape);
  }
  return shape;
};
