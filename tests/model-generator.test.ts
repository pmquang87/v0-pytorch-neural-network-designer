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

    it('generates a Mixture-of-Experts block with a reusable helper class', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 512 }),
                { id: 'moe', type: 'moeNode', data: { d_model: 512, d_ff: 2048, num_experts: 8, top_k: 2, activation: 'silu' } },
            ],
            [{ id: 'e1', source: 'input1', target: 'moe' }],
        ).generateCode();

        // Helper class is emitted once, before the model, and the layer is instantiated + called
        expect(code).toContain('class MixtureOfExperts(nn.Module):');
        expect(code).toContain('self.moe = MixtureOfExperts(d_model=512, d_ff=2048, num_experts=8, top_k=2, activation=nn.SiLU)');
        expect(code).toContain('moe = self.moe(x)');
        // Helper appears before the generated model class
        expect(code.indexOf('class MixtureOfExperts')).toBeLessThan(code.indexOf('class GeneratedModel'));
    });

    it('emits the MoE helper class only once for multiple MoE nodes and maps activation', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 64 }),
                { id: 'moe1', type: 'moeNode', data: { d_model: 64, d_ff: 128, num_experts: 4, top_k: 1, activation: 'gelu' } },
                { id: 'moe2', type: 'moeNode', data: { d_model: 64, d_ff: 128, num_experts: 4, top_k: 2 } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'moe1' },
                { id: 'e2', source: 'moe1', target: 'moe2' },
            ],
        ).generateCode();

        expect(code.match(/class MixtureOfExperts\(nn\.Module\):/g)?.length).toBe(1);
        expect(code).toContain('activation=nn.GELU'); // explicit gelu
        expect(code).toContain('activation=nn.GELU)'); // default when unspecified also falls back to GELU
    });

    it('generates nn.RMSNorm with normalized_shape', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 128 }),
                { id: 'norm', type: 'rmsnormNode', data: { normalized_shape: [128] } },
            ],
            [{ id: 'e1', source: 'input1', target: 'norm' }],
        ).generateCode();

        expect(code).toContain('self.norm = nn.RMSNorm(normalized_shape=(128))');
        expect(code).toContain('norm = self.norm(x)');
    });

    it('omits the GELU approximate arg by default but emits it when set', () => {
        const plain = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 16 }),
                { id: 'act', type: 'geluNode', data: {} },
            ],
            [{ id: 'e1', source: 'input1', target: 'act' }],
        ).generateCode();
        expect(plain).toContain('self.act = nn.GELU()');

        const tanh = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 16 }),
                { id: 'act', type: 'geluNode', data: { approximate: 'tanh' } },
            ],
            [{ id: 'e1', source: 'input1', target: 'act' }],
        ).generateCode();
        expect(tanh).toContain('self.act = nn.GELU(approximate="tanh")');
    });

    it('emits subtraction for addNode with op "-"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 8 }),
                { id: 'bias', type: 'parameterNode', data: { shape: [8] } },
                { id: 'sub1', type: 'addNode', data: { op: '-' } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'sub1' },
                { id: 'e2', source: 'bias', target: 'sub1' },
            ],
        ).generateCode();

        expect(code).toMatch(/sub1 = .*-.*/);
        expect(code).not.toMatch(/sub1 = .*\+.*/);
    });

    it('emits division for multiplyNode with op "/"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 8 }),
                { id: 'scale', type: 'parameterNode', data: { shape: [8] } },
                { id: 'div1', type: 'multiplyNode', data: { op: '/' } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'div1' },
                { id: 'e2', source: 'scale', target: 'div1' },
            ],
        ).generateCode();

        expect(code).toMatch(/div1 = x \/ scale|div1 = scale \/ x/);
        expect(code).not.toMatch(/div1 = .*\*.*/);
    });

    it('emits @ operator for multiplyNode with op "@"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 8 }),
                { id: 'w', type: 'parameterNode', data: { shape: [8, 8] } },
                { id: 'mm', type: 'multiplyNode', data: { op: '@' } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'mm' },
                { id: 'e2', source: 'w', target: 'mm' },
            ],
        ).generateCode();

        expect(code).toMatch(/mm = .* @ .*/);
    });

    it('emits torch.matmul for multiplyNode with op "matmul"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 8 }),
                { id: 'w', type: 'parameterNode', data: { shape: [8, 8] } },
                { id: 'mm', type: 'multiplyNode', data: { op: 'matmul' } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'mm' },
                { id: 'e2', source: 'w', target: 'mm' },
            ],
        ).generateCode();

        expect(code).toMatch(/mm = torch\.matmul\((x, w|w, x)\)/);
    });

    it('nests torch.matmul right-associatively for 3+ inputs', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 8 }),
                { id: 'a', type: 'parameterNode', data: { shape: [8, 8] } },
                { id: 'b', type: 'parameterNode', data: { shape: [8, 8] } },
                { id: 'mm', type: 'multiplyNode', data: { op: 'matmul' } },
            ],
            [
                { id: 'e1', source: 'input1', target: 'mm' },
                { id: 'e2', source: 'a', target: 'mm' },
                { id: 'e3', source: 'b', target: 'mm' },
            ],
        ).generateCode();

        expect(code).toContain('torch.matmul(');
        // right-associative nesting: outer matmul contains an inner matmul
        expect(code).toMatch(/mm = torch\.matmul\([^,]+, torch\.matmul\(/);
    });

    it('emits nn.LogSoftmax for softmaxNode variant "log"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: 'sm', type: 'softmaxNode', data: { variant: 'log', dim: 1 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'sm' }],
        ).generateCode();

        expect(code).toContain('self.sm = nn.LogSoftmax(dim=1)');
        expect(code).toContain('sm = self.sm(x)');
        expect(code).not.toContain('nn.Softmax(');
    });

    it('emits nn.ReLU6 for reluNode variant "relu6"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: 'act', type: 'reluNode', data: { variant: 'relu6' } },
            ],
            [{ id: 'e1', source: 'input1', target: 'act' }],
        ).generateCode();

        expect(code).toContain('self.act = nn.ReLU6()');
    });

    it('emits nn.Dropout2d and nn.Dropout3d for dropout variants', () => {
        const code2d = makeGraph(
            [
                inputNode(),
                { id: 'do', type: 'dropoutNode', data: { variant: 'dropout2d', p: 0.3 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'do' }],
        ).generateCode();
        expect(code2d).toContain('self.do = nn.Dropout2d(p=0.3)');

        const code3d = makeGraph(
            [
                inputNode(),
                { id: 'do', type: 'dropoutNode', data: { variant: 'dropout3d', p: 0.5 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'do' }],
        ).generateCode();
        expect(code3d).toContain('self.do = nn.Dropout3d(p=0.5)');
    });

    it('emits nn.LazyLinear without in_features for linearNode variant "lazy"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 784 }),
                { id: 'fc', type: 'linearNode', data: { variant: 'lazy', out_features: 128, bias: false } },
            ],
            [{ id: 'e1', source: 'input1', target: 'fc' }],
        ).generateCode();

        expect(code).toContain('self.fc = nn.LazyLinear(128, bias=False)');
        expect(code).not.toContain('in_features');
        expect(code).toContain('fc = self.fc(x)');
    });

    it('emits nn.Bilinear for linearNode variant "bilinear"', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 20 }),
                { id: 'bi', type: 'linearNode', data: { variant: 'bilinear', in1_features: 20, in2_features: 30, out_features: 40 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'bi' }],
        ).generateCode();

        expect(code).toContain('self.bi = nn.Bilinear(20, 30, 40)');
    });

    it('behaves as plain nn.Softmax/nn.Linear when no variant is set', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: 'sm', type: 'softmaxNode', data: { dim: 1 } },
            ],
            [{ id: 'e1', source: 'input1', target: 'sm' }],
        ).generateCode();
        expect(code).toContain('self.sm = nn.Softmax(dim=1)');
    });

    it('prefixes numeric node ids so no invalid self.<digit> is emitted', () => {
        const code = makeGraph(
            [
                inputNode('input1', { name: 'x', features: 10 }),
                { id: '123', type: 'linearNode', data: { in_features: 10, out_features: 2 } },
            ],
            [{ id: 'e1', source: 'input1', target: '123' }],
        ).generateCode();

        expect(code).toContain('self.n_123 = nn.Linear');
        expect(code).toContain('n_123 = self.n_123(x)');
        expect(code).not.toContain('self.123');
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
