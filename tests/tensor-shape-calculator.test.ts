import { describe, it, expect } from 'vitest';
import { calculateOutputShape, validateTensorShapes, formatTensorShape } from '../lib/tensor-shape-calculator';

describe('validateTensorShapes', () => {
    it('ignores the batch dimension when validating linear in_features', () => {
        const result = validateTensorShapes(
            'linearNode',
            [{ batch: 32, features: 784 } as any],
            { in_features: 784, out_features: 128 },
        );
        expect(result.isValid).toBe(true);
    });

    it('reports a mismatch for wrong linear in_features', () => {
        const result = validateTensorShapes(
            'linearNode',
            [{ features: 784 }],
            { in_features: 512, out_features: 128 },
        );
        expect(result.isValid).toBe(false);
        expect(result.error).toMatch(/mismatch/i);
    });

    it('flattens multi-dim input when validating linear in_features', () => {
        const result = validateTensorShapes(
            'linearNode',
            [{ channels: 16, height: 4, width: 4 }],
            { in_features: 256, out_features: 10 },
        );
        expect(result.isValid).toBe(true);
    });

    it('accepts broadcast-compatible add inputs', () => {
        const result = validateTensorShapes(
            'addNode',
            [{ channels: 64, height: 8, width: 8 }, { channels: 64, height: 8, width: 8 }],
            {},
        );
        expect(result.isValid).toBe(true);
    });

    it('rejects incompatible add inputs', () => {
        const result = validateTensorShapes(
            'addNode',
            [{ channels: 64, height: 8, width: 8 }, { channels: 32, height: 8, width: 8 }],
            {},
        );
        expect(result.isValid).toBe(false);
    });
});

describe('calculateOutputShape', () => {
    it('preserves channels through MaxPool3d', () => {
        const out = calculateOutputShape(
            'maxpool3dNode',
            [{ channels: 8, depth: 16, height: 16, width: 16 }],
            { kernel_size: 2 },
        );
        expect(out.channels).toBe(8);
        expect(out.depth).toBe(8);
        expect(out.height).toBe(8);
        expect(out.width).toBe(8);
    });

    it('computes constantNode shape from its own data', () => {
        const out = calculateOutputShape('constantNode', [], { channels: 1, height: 4, width: 4 });
        expect(out).toEqual({ channels: 1, height: 4, width: 4 });
    });

    it('upsamples length for ConvTranspose1d', () => {
        const out = calculateOutputShape(
            'convtranspose1dNode',
            [{ channels: 16, length: 10 }],
            { out_channels: 8, kernel_size: 2, stride: 2 },
        );
        expect(out.channels).toBe(8);
        expect(out.length).toBe(20);
    });

    it('upsamples depth/height/width for ConvTranspose3d', () => {
        const out = calculateOutputShape(
            'convtranspose3dNode',
            [{ channels: 16, depth: 4, height: 4, width: 4 }],
            { out_channels: 8, kernel_size: 2, stride: 2 },
        );
        expect(out).toEqual({ channels: 8, depth: 8, height: 8, width: 8 });
    });

    it('applies out_channels and stride for MBConv blocks', () => {
        const out = calculateOutputShape(
            'mbconvNode',
            [{ channels: 32, height: 112, width: 112 }],
            { in_channels: 32, out_channels: 96, kernel_size: 3, stride: 2, expand_ratio: 6 },
        );
        expect(out.channels).toBe(96);
        expect(out.height).toBe(56);
        expect(out.width).toBe(56);
    });

    it('conv2d computes spatial dims with tuple string kernels', () => {
        const out = calculateOutputShape(
            'conv2dNode',
            [{ channels: 3, height: 28, width: 28 }],
            { out_channels: 16, kernel_size: '3,3', stride: 1, padding: 1 },
        );
        expect(out).toEqual({ channels: 16, height: 28, width: 28 });
    });
});

describe('formatTensorShape', () => {
    it('formats canonical dims in order', () => {
        expect(formatTensorShape({ channels: 3, height: 28, width: 28 })).toBe('[3, 28, 28]');
    });

    it('renders dynamic dims as ?', () => {
        expect(formatTensorShape({ channels: 3, height: 'dynamic', width: 28 })).toBe('[3, ?, 28]');
    });

    it('handles empty shapes', () => {
        expect(formatTensorShape({})).toBe('[?]');
        expect(formatTensorShape(undefined)).toBe('[?]');
    });
});
