import { describe, it, expect } from 'vitest';
import { analyzeLayer, analyzeModel } from '../lib/model-analyzer';
import type { TensorShape } from '../lib/tensor-shape-calculator';

const inShape: TensorShape = { channels: 16, height: 8, width: 8 };
const outShape: TensorShape = { channels: 32, height: 8, width: 8 };

describe('analyzeLayer - bias handling', () => {
    it('conv2d includes bias by default', () => {
        const a = analyzeLayer(
            'conv2dNode',
            { in_channels: 16, out_channels: 32, kernel_size: 3 },
            inShape,
            outShape,
        );
        // weights = 16*32*3*3 = 4608, bias = 32 -> 4640
        expect(a.parameters).toBe(4640);
    });

    it('conv2d omits bias when bias === false', () => {
        const a = analyzeLayer(
            'conv2dNode',
            { in_channels: 16, out_channels: 32, kernel_size: 3, bias: false },
            inShape,
            outShape,
        );
        expect(a.parameters).toBe(4608);
    });

    it('conv1d omits bias when bias === false', () => {
        const withBias = analyzeLayer('conv1dNode', { in_channels: 4, out_channels: 8, kernel_size: 3 }, {}, {});
        const noBias = analyzeLayer('conv1dNode', { in_channels: 4, out_channels: 8, kernel_size: 3, bias: false }, {}, {});
        expect(withBias.parameters).toBe(4 * 8 * 3 + 8);
        expect(noBias.parameters).toBe(4 * 8 * 3);
    });

    it('conv3d omits bias when bias === false', () => {
        const noBias = analyzeLayer('conv3dNode', { in_channels: 2, out_channels: 4, kernel_size: 3, bias: false }, {}, {});
        expect(noBias.parameters).toBe(2 * 4 * 3 * 3 * 3);
    });

    it('linear omits bias when bias === false', () => {
        const withBias = analyzeLayer('linearNode', { in_features: 128, out_features: 64 }, {}, {});
        const noBias = analyzeLayer('linearNode', { in_features: 128, out_features: 64, bias: false }, {}, {});
        expect(withBias.parameters).toBe(128 * 64 + 64);
        expect(noBias.parameters).toBe(128 * 64);
    });
});

describe('analyzeLayer - depthwise conv groups', () => {
    it('defaults groups to in_channels when unset', () => {
        const a = analyzeLayer(
            'depthwiseconv2dNode',
            { in_channels: 32, out_channels: 32, kernel_size: 3 },
            inShape,
            outShape,
        );
        // groups = 32 -> weights = 32*32*9/32 = 288, bias = 32 -> 320
        expect(a.parameters).toBe(320);
    });

    it('respects an explicit groups value', () => {
        const a = analyzeLayer(
            'depthwiseconv2dNode',
            { in_channels: 32, out_channels: 32, kernel_size: 3, groups: 1 },
            inShape,
            outShape,
        );
        // groups = 1 -> weights = 32*32*9 = 9216, bias = 32 -> 9248
        expect(a.parameters).toBe(9248);
    });

    it('regular conv2d still defaults groups to 1', () => {
        const a = analyzeLayer(
            'conv2dNode',
            { in_channels: 32, out_channels: 32, kernel_size: 3 },
            inShape,
            outShape,
        );
        expect(a.parameters).toBe(32 * 32 * 9 + 32);
    });
});

describe('analyzeLayer - normalization params', () => {
    it('groupnorm counts 2 * num_channels', () => {
        const a = analyzeLayer('groupnormNode', { num_groups: 4, num_channels: 64 }, {}, {});
        expect(a.parameters).toBe(128);
    });

    it('groupnorm with affine=false has 0 params', () => {
        const a = analyzeLayer('groupnormNode', { num_groups: 4, num_channels: 64, affine: false }, {}, {});
        expect(a.parameters).toBe(0);
    });

    it('instancenorm2d counts 2 * num_features when affine=true', () => {
        const a = analyzeLayer('instancenorm2dNode', { num_features: 48, affine: true }, {}, {});
        expect(a.parameters).toBe(96);
    });

    it('instancenorm defaults to 0 params (affine=false in PyTorch)', () => {
        expect(analyzeLayer('instancenorm1dNode', { num_features: 48 }, {}, {}).parameters).toBe(0);
        expect(analyzeLayer('instancenorm3dNode', { num_features: 10, affine: true }, {}, {}).parameters).toBe(20);
    });
});

describe('analyzeLayer - convtranspose params', () => {
    it('convtranspose2d matches conv2d formula', () => {
        const conv = analyzeLayer('conv2dNode', { in_channels: 8, out_channels: 16, kernel_size: 3 }, {}, {});
        const convT = analyzeLayer('convtranspose2dNode', { in_channels: 8, out_channels: 16, kernel_size: 3 }, {}, {});
        expect(convT.parameters).toBe(conv.parameters);
        expect(convT.parameters).toBe(8 * 16 * 9 + 16);
    });

    it('convtranspose1d/3d produce non-zero params', () => {
        expect(analyzeLayer('convtranspose1dNode', { in_channels: 4, out_channels: 8, kernel_size: 3 }, {}, {}).parameters).toBe(4 * 8 * 3 + 8);
        expect(analyzeLayer('convtranspose3dNode', { in_channels: 2, out_channels: 4, kernel_size: 3 }, {}, {}).parameters).toBe(2 * 4 * 27 + 4);
    });
});

describe('analyzeLayer - embedding', () => {
    it('embedding params = num_embeddings * embedding_dim, 0 FLOPs', () => {
        const a = analyzeLayer('embeddingNode', { num_embeddings: 10000, embedding_dim: 128 }, {}, {});
        expect(a.parameters).toBe(10000 * 128);
        expect(a.flops).toBe(0);
    });
});

describe('analyzeLayer - transformer layers do not silently return 0', () => {
    it('encoder layer has non-zero params', () => {
        const a = analyzeLayer('transformerencoderlayerNode', { d_model: 512, dim_feedforward: 2048 }, {}, {});
        expect(a.parameters).toBeGreaterThan(0);
    });

    it('decoder layer has more params than encoder (cross-attention)', () => {
        const enc = analyzeLayer('transformerencoderlayerNode', { d_model: 512, dim_feedforward: 2048 }, {}, {});
        const dec = analyzeLayer('transformerdecoderlayerNode', { d_model: 512, dim_feedforward: 2048 }, {}, {});
        expect(dec.parameters).toBeGreaterThan(enc.parameters);
    });
});

describe('analyzeLayer - batchnorm trainable vs buffers', () => {
    it('reports 2 * C trainable and 4 * C total', () => {
        const a = analyzeLayer('batchnorm2dNode', { num_features: 64 }, {}, outShape);
        expect(a.parameters).toBe(256); // gamma, beta, running_mean, running_var
        expect(a.trainableParameters).toBe(128); // gamma, beta only
    });

    it('batchnorm1d/3d also report 2 * C trainable', () => {
        expect(analyzeLayer('batchnorm1dNode', { num_features: 32 }, {}, {}).trainableParameters).toBe(64);
        expect(analyzeLayer('batchnorm3dNode', { num_features: 32 }, {}, {}).trainableParameters).toBe(64);
    });
});

describe('analyzeModel - aggregation', () => {
    it('trainableParameters excludes batchnorm buffers', () => {
        const nodes = [
            { id: 'in', type: 'inputNode', data: {} },
            { id: 'bn', type: 'batchnorm2dNode', data: { num_features: 64, outputShape: outShape } },
        ];
        const result = analyzeModel(nodes, []);
        expect(result.totalParameters).toBe(256);
        expect(result.trainableParameters).toBe(128);
    });

    it('trainableParameters equals totalParameters for a plain linear stack', () => {
        const nodes = [
            { id: 'in', type: 'inputNode', data: {} },
            { id: 'fc', type: 'linearNode', data: { in_features: 128, out_features: 64 } },
        ];
        const result = analyzeModel(nodes, []);
        expect(result.trainableParameters).toBe(result.totalParameters);
        expect(result.totalParameters).toBe(128 * 64 + 64);
    });
});
