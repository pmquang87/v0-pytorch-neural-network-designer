import { describe, it, expect } from 'vitest';
import { ModelGenerator } from '../lib/model-generator';
import type { GraphNode, GraphEdge } from '../lib/types';

function makeGraph(nodes: GraphNode[], edges: GraphEdge[]) {
    return new ModelGenerator({ nodes, edges });
}

const inputNode = (id = 'input1', data: any = { name: 'x', channels: 3, height: 28, width: 28 }): GraphNode => ({
    id,
    type: 'inputNode',
    data,
});

describe('ModelGenerator', () => {
    it('generates a simple linear model', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 784 }),
                { id: 'fc1', type: 'linearNode', data: { in_features: 784, out_features: 128 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'fc1' }],
        ).generateCode();

        expect(code).toContain('self.fc1 = nn.Linear(in_features=784, out_features=128)');
        expect(code).toContain('fc1 = self.fc1(x)');
        expect(code).toContain('return fc1');
    });

    it('emits Python booleans (True/False), not JS booleans', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 784 }),
                { id: 'fc1', type: 'linearNode', data: { in_features: 784, out_features: 128, bias: false } },
            ],
            [{ id: 'e1', source: 'input1', target: 'fc1' }],
        ).generateCode();

        expect(code).toContain('bias=False');
        expect(code).not.toContain('bias=false');
    });

    it('unpacks LSTM tuple outputs in the forward pass', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', sequence: 10, features: 32 }),
                { id: 'lstm1', type: 'lstmNode', data: { input_size: 32, hidden_size: 64, batch_first: true } },
            ],
            [{ id: 'e1', source: 'input1', target: 'lstm1' }],
        ).generateCode();

        expect(code).toContain('lstm1, _ = self.lstm1(x)');
        expect(code).toContain('batch_first=True');
    });

    it('calls MultiheadAttention with query/key/value and unpacks the tuple', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', sequence: 10, features: 128 }),
                { id: 'attn1', type: 'multiheadattentionNode', data: { embed_dim: 128, num_heads: 8 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'attn1' }],
        ).generateCode();

        expect(code).toContain('attn1, _ = self.attn1(x, x, x)');
    });

    it('parses string targetShape for reshape nodes into a tuple literal', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', channels: 1, height: 28, width: 28 }),
                { id: 'reshape1', type: 'reshapeNode', data: { targetShape: '[-1, 784]' } },
            ],
            [{ id: 'e1', source: 'input1', target: 'reshape1' }],
        ).generateCode();

        expect(code).toContain('reshape1 = x.view(x.size(0), *(-1, 784))');
        expect(code).not.toContain('*"');
    });

    it('adds groups=in_channels for depthwise convolutions', () => {
        const code = makeGraph(
            [
                inputNode(),
                { id: 'dw1', type: 'depthwiseconv2dNode', data: { in_channels: 32, out_channels: 32, kernel_size: 3 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'dw1' }],
        ).generateCode();

        expect(code).toContain('groups=32');
    });

    it('expands separable convolutions into depthwise + pointwise', () => {
        const code = makeGraph(
            [
                inputNode(),
                { id: 'sep1', type: 'separableconv2dNode', data: { in_channels: 3, out_channels: 64, kernel_size: 3 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'sep1' }],
        ).generateCode();

        expect(code).toContain('nn.Sequential');
        expect(code).toContain('groups=3');
        expect(code).toContain('kernel_size=1');
        expect(code).toContain('sep1 = self.sep1(x)');
    });

    it('does not call undefined layers for unsupported node types', () => {
        const code = makeGraph(
            [
                inputNode(),
                { id: 'mystery1', type: 'someUnknownNode', data: {} },
            ],
            [{ id: 'e1', source: 'input1', target: 'mystery1' }],
        ).generateCode();

        expect(code).not.toContain('self.mystery1(');
        expect(code).toContain("# TODO: 'someUnknownNode' is not supported");
    });

    it('emits tuple literals for comma-separated string parameters', () => {
        const code = makeGraph(
            [
                inputNode(),
                { id: 'conv1', type: 'conv2dNode', data: { in_channels: 3, out_channels: 16, kernel_size: '3, 3' } },
            ],
            [{ id: 'e1', source: 'input1', target: 'conv1' }],
        ).generateCode();

        expect(code).toContain('kernel_size=(3, 3)');
        expect(code).not.toContain('kernel_size="');
    });

    it('resolves the return variable through the output node', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: 'fc1', type: 'linearNode', data: { in_features: 10, out_features: 2 } },
                { id: 'out1', type: 'outputNode', data: {} },
            ],
            [
                { id: 'e1', source: 'input1', target: 'fc1' },
                { id: 'e2', source: 'fc1', target: 'out1' },
            ],
        ).generateCode();

        expect(code).toContain('return fc1');
    });

    it('returns the named input variable when the output node is fed directly by an input', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: 'out1', type: 'outputNode', data: {} },
            ],
            [{ id: 'e1', source: 'input1', target: 'out1' }],
        ).generateCode();

        expect(code).toContain('return x');
    });

    it('emits nn.Parameter for parameter source nodes', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', sequence: 4, features: 8 }),
                { id: 'pos_emb', type: 'parameterNode', data: { shape: [1, 4, 8] } },
                { id: 'add1', type: 'addNode', data: {} },
            ],
            [
                { id: 'e1', source: 'input1', target: 'add1' },
                { id: 'e2', source: 'pos_emb', target: 'add1' },
            ],
        ).generateCode();

        expect(code).toContain('self.pos_emb = nn.Parameter(torch.randn(1, 4, 8))');
        expect(code).toContain('pos_emb = self.pos_emb');
        expect(code).toMatch(/add1 = .*\+.*/);
    });

    it('emits torch.zeros for constant source nodes', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', channels: 1, height: 4, width: 4 }),
                { id: 'const1', type: 'constantNode', data: { channels: 1, height: 4, width: 4 } },
                { id: 'add1', type: 'addNode', data: {} },
            ],
            [
                { id: 'e1', source: 'input1', target: 'add1' },
                { id: 'e2', source: 'const1', target: 'add1' },
            ],
        ).generateCode();

        expect(code).toContain('const1 = torch.zeros(1, 1, 4, 4)');
    });

    it('rejects graphs without an input node', () => {
        const generator = makeGraph(
            [{ id: 'fc1', type: 'linearNode', data: { in_features: 4, out_features: 2 } }],
            [],
        );
        expect(() => generator.generateCode()).toThrow(/input node/i);
    });

    it('rejects cyclic graphs', () => {
        const generator = makeGraph(
            [
                inputNode(),
                { id: 'a', type: 'linearNode', data: { in_features: 4, out_features: 4 } },
                { id: 'b', type: 'linearNode', data: { in_features: 4, out_features: 4 } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'a' },
                { id: 'e2', source: 'a', target: 'b' },
                { id: 'e3', source: 'b', target: 'a' },
            ],
        );
        expect(() => generator.generateCode()).toThrow(/cycle/i);
    });
});
