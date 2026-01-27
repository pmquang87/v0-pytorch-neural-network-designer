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
});
