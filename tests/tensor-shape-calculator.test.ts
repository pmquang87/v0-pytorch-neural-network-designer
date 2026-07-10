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

    it('validates linear in_features against the last dimension only (issue #5)', () => {
        // Input [3, 30, 28] -> nn.Linear expects in_features=28, not 3*30*28
        const result = validateTensorShapes(
            'linearNode',
            [{ channels: 3, height: 30, width: 28 }],
            { in_features: 28, out_features: 64 },
        );
        expect(result.isValid).toBe(true);
    });

    it('rejects linear in_features equal to the flattened product of a multi-dim input', () => {
        const result = validateTensorShapes(
            'linearNode',
            [{ channels: 3, height: 30, width: 28 }],
            { in_features: 2520, out_features: 64 },
        );
        expect(result.isValid).toBe(false);
        expect(result.error).toMatch(/last dimension/i);
    });

    it('accepts a dynamic last dimension for linear layers', () => {
        const result = validateTensorShapes(
            'linearNode',
            [{ sequence: 10, features: 'dynamic' }],
            { in_features: 128, out_features: 64 },
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
    it('linear preserves leading dimensions and replaces the last one (issue #5)', () => {
        // Input [3, 30, 28] -> Linear(28, 64) -> [3, 30, 64]
        const out = calculateOutputShape(
            'linearNode',
            [{ channels: 3, height: 30, width: 28 }],
            { in_features: 28, out_features: 64 },
        );
        expect(out).toEqual({ channels: 3, height: 30, width: 64 });
    });

    it('linear on a flat feature vector replaces features', () => {
        const out = calculateOutputShape(
            'linearNode',
            [{ features: 784 }],
            { in_features: 784, out_features: 128 },
        );
        expect(out).toEqual({ features: 128 });
    });

    it('linear with no input shape falls back to out_features', () => {
        const out = calculateOutputShape('linearNode', [{}], { in_features: 10, out_features: 5 });
        expect(out).toEqual({ features: 5 });
    });

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

    it('doubles hidden_size features for a bidirectional LSTM', () => {
        const out = calculateOutputShape(
            'lstmNode',
            [{ sequence: 10, features: 64 }],
            { input_size: 64, hidden_size: 128, bidirectional: true },
        );
        expect(out).toEqual({ sequence: 10, features: 256 });
    });

    it('keeps hidden_size features for a unidirectional GRU', () => {
        const out = calculateOutputShape(
            'gruNode',
            [{ sequence: 10, features: 64 }],
            { input_size: 64, hidden_size: 128 },
        );
        expect(out).toEqual({ sequence: 10, features: 128 });
    });

    it('maxpool2d with a tuple string kernel yields a concrete shape (not dynamic)', () => {
        const out = calculateOutputShape(
            'maxpool2dNode',
            [{ channels: 8, height: 32, width: 32 }],
            { kernel_size: '2,2' },
        );
        expect(out).toEqual({ channels: 8, height: 16, width: 16 });
        expect(out.height).not.toBe('dynamic');
        expect(out.width).not.toBe('dynamic');
    });

    it('maxpool2d with an array tuple kernel/stride is concrete', () => {
        const out = calculateOutputShape(
            'maxpool2dNode',
            [{ channels: 4, height: 30, width: 30 }],
            { kernel_size: [3, 3], stride: [3, 3] },
        );
        expect(out).toEqual({ channels: 4, height: 10, width: 10 });
    });

    it('lppool2d with a tuple string kernel is concrete', () => {
        const out = calculateOutputShape(
            'lppool2dNode',
            [{ channels: 4, height: 16, width: 16 }],
            { kernel_size: '2,2' },
        );
        expect(out).toEqual({ channels: 4, height: 8, width: 8 });
    });

    it("conv2d padding='same' preserves H/W", () => {
        const out = calculateOutputShape(
            'conv2dNode',
            [{ channels: 3, height: 28, width: 28 }],
            { out_channels: 16, kernel_size: 5, stride: 1, padding: 'same' },
        );
        expect(out).toEqual({ channels: 16, height: 28, width: 28 });
    });

    it("conv2d padding='valid' means no padding", () => {
        const out = calculateOutputShape(
            'conv2dNode',
            [{ channels: 3, height: 28, width: 28 }],
            { out_channels: 16, kernel_size: 3, stride: 1, padding: 'valid' },
        );
        expect(out).toEqual({ channels: 16, height: 26, width: 26 });
    });

    it("conv1d padding='same' preserves length", () => {
        const out = calculateOutputShape(
            'conv1dNode',
            [{ channels: 3, length: 100 }],
            { out_channels: 16, kernel_size: 5, stride: 1, padding: 'same' },
        );
        expect(out).toEqual({ channels: 16, length: 100 });
    });

    it('conv1d parses tuple string kernel/padding without NaN', () => {
        const out = calculateOutputShape(
            'conv1dNode',
            [{ channels: 3, length: 100 }],
            { out_channels: 8, kernel_size: '3', stride: 1, padding: '1' },
        );
        expect(out).toEqual({ channels: 8, length: 100 });
    });

    it("conv3d padding='same' preserves D/H/W", () => {
        const out = calculateOutputShape(
            'conv3dNode',
            [{ channels: 2, depth: 8, height: 8, width: 8 }],
            { out_channels: 4, kernel_size: 3, stride: 1, padding: 'same' },
        );
        expect(out).toEqual({ channels: 4, depth: 8, height: 8, width: 8 });
    });

    it('adaptiveavgpool2d accepts a scalar output_size', () => {
        const out = calculateOutputShape(
            'adaptiveavgpool2dNode',
            [{ channels: 16, height: 32, width: 32 }],
            { output_size: 7 },
        );
        expect(out).toEqual({ channels: 16, height: 7, width: 7 });
    });

    it('adaptiveavgpool2d accepts an array output_size', () => {
        const out = calculateOutputShape(
            'adaptiveavgpool2dNode',
            [{ channels: 16, height: 32, width: 32 }],
            { output_size: [4, 8] },
        );
        expect(out).toEqual({ channels: 16, height: 4, width: 8 });
    });

    it('adaptivemaxpool2d mirrors the avg variant', () => {
        const out = calculateOutputShape(
            'adaptivemaxpool2dNode',
            [{ channels: 8, height: 20, width: 20 }],
            { output_size: 1 },
        );
        expect(out).toEqual({ channels: 8, height: 1, width: 1 });
    });

    it('adaptiveavgpool1d produces a length dimension', () => {
        const out = calculateOutputShape(
            'adaptiveavgpool1dNode',
            [{ channels: 8, length: 100 }],
            { output_size: 5 },
        );
        expect(out).toEqual({ channels: 8, length: 5 });
    });

    it('adaptiveavgpool3d produces D/H/W dimensions', () => {
        const out = calculateOutputShape(
            'adaptiveavgpool3dNode',
            [{ channels: 8, depth: 16, height: 16, width: 16 }],
            { output_size: [2, 2, 2] },
        );
        expect(out).toEqual({ channels: 8, depth: 2, height: 2, width: 2 });
    });

    it('embedding appends embedding_dim as features, preserving the sequence', () => {
        const out = calculateOutputShape(
            'embeddingNode',
            [{ sequence: 32 }],
            { num_embeddings: 1000, embedding_dim: 64 },
        );
        expect(out).toEqual({ sequence: 32, features: 64 });
    });

    it('embedding treats a flat features dim as the sequence of ids', () => {
        const out = calculateOutputShape(
            'embeddingNode',
            [{ features: 32 }],
            { num_embeddings: 1000, embedding_dim: 64 },
        );
        expect(out).toEqual({ sequence: 32, features: 64 });
    });

    it('embedding with no input dims yields just features', () => {
        const out = calculateOutputShape(
            'embeddingNode',
            [{}],
            { num_embeddings: 1000, embedding_dim: 128 },
        );
        expect(out).toEqual({ features: 128 });
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
