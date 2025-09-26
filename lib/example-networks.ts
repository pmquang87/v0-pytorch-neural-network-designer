import type { Node, Edge } from "@xyflow/react";

export interface ExampleNetwork {
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

export interface ExampleNetworkMetadata {
  name: string;
  filename: string;
}

export const EXAMPLE_NETWORKS_METADATA: ExampleNetworkMetadata[] = [
  { name: "T5", filename: "T5.json" },
  { name: "GRU", filename: "GRU.json" },
  { name: "RNN", filename: "RNN.json" },
  { name: "BART", filename: "BART_full.json" },
  { name: "BART (Simplified)", filename: "BART_simplified.json" },
  { name: "BERT", filename: "BERT.json" },
  { name: "GPT2", filename: "GPT2.json" },
  { name: "Mamba", filename: "Mamba.json" },
  { name: "SENet", filename: "SENet.json" },
  { name: "U-Net", filename: "U-Net.json" },
  { name: "YOLOv1", filename: "YOLOv1.json" },
  { name: "YOLOv3", filename: "YOLOv3.json" },
  { name: "AlexNet", filename: "AlexNet.json" },
  { name: "LeNet-5", filename: "LeNet-5.json" },
  { name: "RNNCell", filename: "RNNCell.json" },
  { name: "U-Net++", filename: "U-Net++.json" },
  { name: "YOLOv5s", filename: "YOLOv5s.json" },
  { name: "YOLOv8n", filename: "YOLOv8n.json" },
  { name: "CoAtNet-7", filename: "CoAtNet7.json" },
  { name: "CoAtNet-7 (Simplified)", filename: "CoAtNet7-simplified.json" },
  { name: "GRU-Node", filename: "GRU-Node.json" },
  { name: "YOLO-NAS", filename: "YOLO-NAS.json" },
  { name: "FullSENet", filename: "FullSENet.json" },
  { name: "GPT2-full", filename: "GPT2-full.json" },
  { name: "LSTM-Node", filename: "LSTM-Node.json" },
  { name: "ResNet-18", filename: "ResNet-18.json" },
  { name: "ResNet-50", filename: "ResNet-50.json" },
  { name: "ResNet-101", filename: "ResNet-101.json" },
  { name: "Simple-MLP", filename: "Simple-MLP.json" },
  { name: "Autoencoder", filename: "Autoencoder.json" },
  { name: "MobileNetV1", filename: "MobileNetV1.json" },
  { name: "Simple-YOLO", filename: "Simple-YOLO.json" },
  { name: "Transformer", filename: "Transformer.json" },
  { name: "DenseNet-121", filename: "DenseNet-121.json" },
  { name: "LSTM-Network", filename: "LSTM-Network.json" },
  { name: "ResNet-Block", filename: "ResNet-Block.json" },
  { name: "VGG-16-Block", filename: "VGG-16-Block.json" },
  { name: "VGG-16", filename: "VGG-16.json" },
  { name: "VGG-19", filename: "VGG-19.json" },
  { name: "GAN-Generator", filename: "GAN-Generator.json" },
  { name: "U-Net-Encoder", filename: "U-Net-Encoder.json" },
  { name: "DenseNet-Block", filename: "DenseNet-Block.json" },
  { name: "EfficientNet-B0", filename: "EfficientNet-B0-complete.json" },
  { name: "MobileNet-Block", filename: "MobileNet-Block.json" },
  { name: "Siamese-Network", filename: "Siamese-Network.json" },
  { name: "Style Transfer", filename: "style-transfer-example.json" },
  { name: "Inception-Module", filename: "Inception-Module.json" },
  { name: "Expanded-RNN-Cell", filename: "Expanded-RNN-Cell.json" },
  { name: "MobileNetV2-Block", filename: "MobileNetV2-Block.json" },
  { name: "Transformer-Block", filename: "Transformer-Block.json" },
  { name: "StyleGAN-Generator", filename: "StyleGAN-Generator.json" },
  { name: "Transformer-Decoder", filename: "Transformer-Decoder.json" },
  { name: "Transformer-Encoder", filename: "Transformer-Encoder.json" },
  { name: "AmoebaNet-B-Normal-Cell", filename: "AmoebaNet-B-Normal-Cell.json" },
  { name: "AmoebaNet-B-Reduction-Cell", filename: "AmoebaNet-B-Reduction-Cell.json" },
  { name: "Vision-Transformer-(ViT-Base-16)", filename: "Vision-Transformer-(ViT-Base-16).json" },
].sort((a, b) => a.name.localeCompare(b.name));