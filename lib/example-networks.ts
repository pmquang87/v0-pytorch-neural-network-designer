import type { Node, Edge } from "@xyflow/react";

export interface ExampleNetwork {
  name: string;
  description: string;
  nodes: Node[];
  edges: Edge[];
}

import AlexNet from "./examples/AlexNet.json";
import Autoencoder from "./examples/Autoencoder.json";
import BART from "./examples/BART.json";
import BERT from "./examples/BERT.json";
import DenseNetBlock from "./examples/DenseNet-Block.json";
import DenseNet121 from "./examples/DenseNet-121.json";
import DenseNet169 from "./examples/DenseNet-169.json";
import DenseNet201 from "./examples/DenseNet-201.json";
import DenseNet264 from "./examples/DenseNet-264.json";
import EfficientNetB0 from "./examples/EfficientNet-B0.json";
import ExpandedRNNCell from "./examples/Expanded-RNN-Cell.json";
import UNet from "./examples/U-Net.json";
import UNetPlusPlus from "./examples/U-Net++.json";
import GANGenerator from "./examples/GAN-Generator.json";
import GRU from "./examples/GRU.json";
import GRUNode from "./examples/GRU-Node.json";
import InceptionModule from "./examples/Inception-Module.json";
import LeNet5 from "./examples/LeNet-5.json";
import LSTMNetwork from "./examples/LSTM-Network.json";
import LSTMNode from "./examples/LSTM-Node.json";
import MaskedMultiheadAttentionNode from "./examples/Masked-MultiheadAttention-Node.json";
import MobileNetBlock from "./examples/MobileNet-Block.json";
import MobileNetV1 from "./examples/MobileNetV1.json";
import MobileNetV2Block from "./examples/MobileNetV2-Block.json";
import MultiheadAttentionNode from "./examples/MultiheadAttention-Node.json";
import ResNetBlock from "./examples/ResNet-Block.json";
import ResNet50 from "./examples/ResNet-50.json";
import ResNet101 from "./examples/ResNet-101.json";
import RNN from "./examples/RNN.json";
import SiameseNetwork from "./examples/Siamese-Network.json";
import SimpleMLP from "./examples/Simple-MLP.json";
import SimpleYOLO from "./examples/Simple-YOLO.json";
import StyleGANGenerator from "./examples/StyleGAN-Generator.json";
import T5 from "./examples/T5.json";
import Transformer from "./examples/Transformer.json";
import TransformerBlock from "./examples/Transformer-Block.json";
import TransformerDecoder from "./examples/Transformer-Decoder.json";
import TransformerEncoder from "./examples/Transformer-Encoder.json";
import UNetEncoder from "./examples/U-Net-Encoder.json";
import VGG16Block from "./examples/VGG-16-Block.json";
import VisionTransformer from "./examples/Vision-Transformer-(ViT-Base-16).json";
import YOLONAS from "./examples/YOLO-NAS.json";
import YOLOv1 from "./examples/YOLOv1.json";
import YOLOv3 from "./examples/YOLOv3.json";
import YOLOv5s from "./examples/YOLOv5s.json";
import YOLOv8n from "./examples/YOLOv8n.json";

export const EXAMPLE_NETWORKS: ExampleNetwork[] = [
  AlexNet,
  Autoencoder,
  BART,
  BERT,
  DenseNetBlock,
  DenseNet121,
  DenseNet169,
  DenseNet201,
  DenseNet264,
  EfficientNetB0,
  ExpandedRNNCell,
  UNet,
  UNetPlusPlus,
  GANGenerator,
  GRU,
  GRUNode,
  InceptionModule,
  LeNet5,
  LSTMNetwork,
  LSTMNode,
  MaskedMultiheadAttentionNode,
  MobileNetBlock,
  MobileNetV1,
  MobileNetV2Block,
  MultiheadAttentionNode,
  ResNetBlock,
  ResNet50,
  ResNet101,
  RNN,
  SiameseNetwork,
  SimpleMLP,
  SimpleYOLO,
  StyleGANGenerator,
  T5,
  Transformer,
  TransformerBlock,
  TransformerDecoder,
  TransformerEncoder,
  UNetEncoder,
  VGG16Block,
  VisionTransformer,
  YOLONAS,
  YOLOv1,
  YOLOv3,
  YOLOv5s,
  YOLOv8n,
].sort((a, b) => a.name.localeCompare(b.name));
