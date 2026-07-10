import { describe, it, expect, beforeEach } from 'vitest';
import { ModelValidator } from '../lib/model-validator';

describe('ModelValidator', () => {
    let validator: ModelValidator;

    beforeEach(() => {
        validator = new ModelValidator();
    });

    describe('testShapeCompatibility', () => {
        it('should accept basic matching shapes', () => {
            const shape1 = { batch: 32, features: 784 };
            const shape2 = { batch: 32, features: 784 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(true);
        });

        it('should reject different feature dimensions', () => {
            const shape1 = { batch: 32, features: 784 };
            const shape2 = { batch: 32, features: 512 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(false);
        });

        it('should accept dynamic dimensions with any value', () => {
            const shape1 = { batch: 'dynamic', features: 784 };
            const shape2 = { batch: 32, features: 784 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(true);
        });

        it('should accept Linear layer with matching in_features', () => {
            const shape1 = { batch: 32, features: 784 };
            const shape2 = { batch: 32, in_features: 784, out_features: 128 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(true);
        });

        it('should reject Linear layer with mismatched in_features', () => {
            const shape1 = { batch: 32, features: 784 };
            const shape2 = { batch: 32, in_features: 512, out_features: 128 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(false);
        });

        it('should accept Conv layer with matching in_channels', () => {
            const shape1 = { batch: 32, channels: 3, height: 224, width: 224 };
            const shape2 = { batch: 32, in_channels: 3, out_channels: 64, kernel_size: 3 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(true);
        });

        it('should reject Conv layer with mismatched in_channels', () => {
            const shape1 = { batch: 32, channels: 3, height: 224, width: 224 };
            const shape2 = { batch: 32, in_channels: 1, out_channels: 64, kernel_size: 3 };
            expect(validator.testShapeCompatibility(shape1, shape2)).toBe(false);
        });
    });

    describe('validateModel', () => {
        it('should validate a simple valid model', () => {
            const validNodes = [
                {
                    id: 'input-1',
                    type: 'inputNode',
                    position: { x: 0, y: 0 },
                    data: {
                        batch_size: 32,
                        features: 784,
                        outputShape: { batch: 32, features: 784 }
                    }
                },
                {
                    id: 'linear-1',
                    type: 'linearNode',
                    position: { x: 200, y: 0 },
                    data: {
                        in_features: 784,
                        out_features: 128,
                        inputShape: { batch: 32, features: 784 },
                        outputShape: { batch: 32, features: 128 }
                    }
                }
            ];

            const validEdges = [
                {
                    id: 'edge-1',
                    source: 'input-1',
                    target: 'linear-1'
                }
            ];

            const result = validator.validateModel(validNodes, validEdges);
            expect(result.isValid).toBe(true);
            expect(result.errors).toHaveLength(0);
        });

        it('should detect shape mismatch in invalid model', () => {
            const invalidNodes = [
                {
                    id: 'input-1',
                    type: 'inputNode',
                    position: { x: 0, y: 0 },
                    data: {
                        batch_size: 32,
                        features: 784,
                        outputShape: { batch: 32, features: 784 }
                    }
                },
                {
                    id: 'linear-1',
                    type: 'linearNode',
                    position: { x: 200, y: 0 },
                    data: {
                        in_features: 512, // Mismatch with input features
                        out_features: 128,
                        inputShape: { batch: 32, features: 784 },
                        outputShape: { batch: 32, features: 128 }
                    }
                }
            ];

            const invalidEdges = [
                {
                    id: 'edge-1',
                    source: 'input-1',
                    target: 'linear-1'
                }
            ];

            const result = validator.validateModel(invalidNodes, invalidEdges);
            expect(result.isValid).toBe(false);
            expect(result.errors.length).toBeGreaterThan(0);
            expect(result.errors.some(err => err.includes('mismatch'))).toBe(true);
        });
    });

    describe('actionable shape-mismatch messages', () => {
        it('Linear in_features mismatch states expected vs actual and suggests a fix (BERT/T5 class)', () => {
            const nodes = [
                {
                    id: 'input-1',
                    type: 'inputNode',
                    position: { x: 0, y: 0 },
                    data: { features: 393216, outputShape: { features: 393216 } },
                },
                {
                    id: 'linear-1',
                    type: 'linearNode',
                    position: { x: 200, y: 0 },
                    data: { in_features: 768, out_features: 768 },
                },
            ];
            const edges = [{ id: 'e1', source: 'input-1', target: 'linear-1' }];

            const result = validator.validateModel(nodes, edges);
            expect(result.isValid).toBe(false);
            const msg = result.errors.find(e => e.includes('in_features')) ?? '';
            // expected vs actual numbers
            expect(msg).toMatch(/in_features=768/);
            expect(msg).toMatch(/last dimension is 393216/);
            // concrete fix suggestions
            expect(msg).toMatch(/Flatten/);
            expect(msg).toMatch(/in_features=393216/);
            expect(msg).toMatch(/single token\/position/);
        });

        it('Conv2d in_channels mismatch names both channel counts and the fix', () => {
            const nodes = [
                {
                    id: 'input-1',
                    type: 'inputNode',
                    position: { x: 0, y: 0 },
                    data: { channels: 32, height: 8, width: 8, outputShape: { channels: 32, height: 8, width: 8 } },
                },
                {
                    id: 'conv-1',
                    type: 'conv2dNode',
                    position: { x: 200, y: 0 },
                    data: { in_channels: 64, out_channels: 64, kernel_size: 3 },
                },
            ];
            const edges = [{ id: 'e1', source: 'input-1', target: 'conv-1' }];

            const result = validator.validateModel(nodes, edges);
            expect(result.isValid).toBe(false);
            const msg = result.errors.find(e => e.includes('in_channels')) ?? '';
            expect(msg).toMatch(/Conv2d expects in_channels=64/);
            expect(msg).toMatch(/incoming tensor has 32 channels/);
            expect(msg).toMatch(/Set in_channels=32/);
        });

        it('Add with incompatible inputs names both shapes and the offending dim', () => {
            const nodes = [
                { id: 'a', type: 'inputNode', position: { x: 0, y: 0 }, data: { features: 128, outputShape: { features: 128 } } },
                { id: 'b', type: 'inputNode', position: { x: 0, y: 100 }, data: { features: 256, outputShape: { features: 256 } } },
                { id: 'add', type: 'addNode', position: { x: 200, y: 50 }, data: { num_inputs: 2 } },
            ];
            const edges = [
                { id: 'e1', source: 'a', target: 'add', targetHandle: 'input1' },
                { id: 'e2', source: 'b', target: 'add', targetHandle: 'input2' },
            ];

            const result = validator.validateModel(nodes, edges);
            expect(result.isValid).toBe(false);
            const msg = result.errors.find(e => e.includes('Add error')) ?? '';
            expect(msg).toMatch(/\[128\]/);
            expect(msg).toMatch(/\[256\]/);
            expect(msg).toMatch(/dimension 'features' differs \(128 vs 256\)/);
            expect(msg).toMatch(/broadcastable/);
        });

        it('Concatenate with mismatched non-concat dim names both shapes and the dim', () => {
            const nodes = [
                { id: 'a', type: 'inputNode', position: { x: 0, y: 0 }, data: { sequence: 10, features: 128, outputShape: { sequence: 10, features: 128 } } },
                { id: 'b', type: 'inputNode', position: { x: 0, y: 100 }, data: { sequence: 20, features: 128, outputShape: { sequence: 20, features: 128 } } },
                { id: 'cat', type: 'concatenateNode', position: { x: 200, y: 50 }, data: { num_inputs: 2, dim: 2 } },
            ];
            const edges = [
                { id: 'e1', source: 'a', target: 'cat', targetHandle: 'input1' },
                { id: 'e2', source: 'b', target: 'cat', targetHandle: 'input2' },
            ];

            const result = validator.validateModel(nodes, edges);
            expect(result.isValid).toBe(false);
            const msg = result.errors.find(e => e.includes('Concatenate error')) ?? '';
            expect(msg).toMatch(/\[10, 128\]/);
            expect(msg).toMatch(/\[20, 128\]/);
            expect(msg).toMatch(/non-concatenated dimensions must match/);
            expect(msg).toMatch(/dimension 1 differs \(10 vs 20\)/);
        });
    });

    describe('missing required parameters', () => {
        it('names the missing param, its layer, and a sensible default', () => {
            const nodes = [
                {
                    id: 'conv-1',
                    type: 'conv2dNode',
                    position: { x: 0, y: 0 },
                    data: { in_channels: 3, kernel_size: 3 }, // out_channels missing
                },
            ];

            const result = validator.validateModel(nodes, []);
            expect(result.isValid).toBe(false);
            const msg = result.errors.find(e => e.includes('out_channels')) ?? '';
            expect(msg).toMatch(/Conv2d node/);
            expect(msg).toMatch(/out_channels/);
            expect(msg).toMatch(/number of output feature maps/);
            expect(msg).toMatch(/e\.g\. 64/);
        });

        it('Linear reports each missing param with a default suggestion', () => {
            const nodes = [
                { id: 'lin-1', type: 'linearNode', position: { x: 0, y: 0 }, data: {} },
            ];
            const result = validator.validateModel(nodes, []);
            const msg = result.errors.find(e => e.includes('Linear node')) ?? '';
            expect(msg).toMatch(/in_features/);
            expect(msg).toMatch(/out_features/);
        });
    });
});
