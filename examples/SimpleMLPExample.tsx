import React, { useState } from 'react';
import { Button } from "../components/ui/button";
import { useModelValidation } from "../lib/model-validator";
import { Alert, AlertTitle, AlertDescription } from "../components/ui/alert";
import { Info, AlertCircle } from "lucide-react";

// Example nodes and edges for a simple MLP model
const initialNodes = [
  {
    id: 'input-1',
    type: 'inputNode',
    position: { x: 100, y: 200 },
    data: {
      name: 'Input Layer',
      batch_size: 32,
      features: 784,  // Input features (e.g., 28x28 MNIST image flattened)
      outputShape: {
        batch: 32,
        features: 784
      }
    }
  },
  {
    id: 'linear-1',
    type: 'linearNode',
    position: { x: 300, y: 200 },
    data: {
      name: 'Hidden Layer',
      in_features: 784,  // Must match input features
      out_features: 128,
      inputShape: {
        batch: 32,
        features: 784
      },
      outputShape: {
        batch: 32,
        features: 128
      }
    }
  },
  {
    id: 'relu-1',
    type: 'reluNode',
    position: { x: 500, y: 200 },
    data: {
      name: 'ReLU Activation',
      inputShape: {
        batch: 32,
        features: 128
      },
      outputShape: {
        batch: 32,
        features: 128
      }
    }
  },
  {
    id: 'linear-2',
    type: 'linearNode',
    position: { x: 700, y: 200 },
    data: {
      name: 'Output Layer',
      in_features: 128,  // Must match previous layer's output features
      out_features: 10,  // 10 classes for digit classification
      inputShape: {
        batch: 32,
        features: 128
      },
      outputShape: {
        batch: 32,
        features: 10
      }
    }
  }
];

const initialEdges = [
  {
    id: 'edge-input-linear',
    source: 'input-1',
    target: 'linear-1',
    type: 'default'
  },
  {
    id: 'edge-linear-relu',
    source: 'linear-1',
    target: 'relu-1',
    type: 'default'
  },
  {
    id: 'edge-relu-output',
    source: 'relu-1',
    target: 'linear-2',
    type: 'default'
  }
];

export default function SimpleMLPExample() {
  const [nodes, setNodes] = useState(initialNodes);
  const [edges, setEdges] = useState(initialEdges);
  const { validateModel } = useModelValidation();
  const [validationResult, setValidationResult] = useState(null);

  const handleValidate = () => {
    const result = validateModel(nodes, edges);
    setValidationResult(result);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">Simple MLP Example</h1>
        <Button onClick={handleValidate} className="flex items-center">
          <Info className="h-4 w-4 mr-2" />
          Validate Model
        </Button>
      </div>

      {validationResult && (
        <div className="mb-6">
          {validationResult.isValid ? (
            <Alert className="bg-green-50 border-green-200">
              <AlertTitle className="flex items-center text-green-800">
                <Info className="h-4 w-4 mr-2" />
                Model Validation Successful
              </AlertTitle>
              <AlertDescription className="text-green-700">
                Your model has no errors and is ready to be used.
              </AlertDescription>
            </Alert>
          ) : (
            <Alert className="bg-red-50 border-red-200">
              <AlertTitle className="flex items-center text-red-800">
                <AlertCircle className="h-4 w-4 mr-2" />
                Model Validation Failed
              </AlertTitle>
              <AlertDescription>
                <div className="text-red-700">
                  <h3 className="font-semibold mt-2">Errors ({validationResult.errors.length})</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    {validationResult.errors.map((error, index) => (
                      <li key={index}>{error}</li>
                    ))}
                  </ul>

                  {validationResult.warnings.length > 0 && (
                    <>
                      <h3 className="font-semibold mt-4">Warnings ({validationResult.warnings.length})</h3>
                      <ul className="list-disc pl-5 space-y-1">
                        {validationResult.warnings.map((warning, index) => (
                          <li key={index}>{warning}</li>
                        ))}
                      </ul>
                    </>
                  )}
                </div>
              </AlertDescription>
            </Alert>
          )}
        </div>
      )}

      <div className="bg-gray-50 border rounded-lg p-4">
        <h2 className="text-xl font-semibold mb-2">Model Structure</h2>
        <p className="mb-4">This is a simple MLP model for image classification:</p>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {initialNodes.map(node => (
            <div key={node.id} className="border rounded-lg p-4 bg-white shadow-sm">
              <h3 className="font-semibold text-primary">{node.data.name}</h3>
              <div className="text-sm mt-2">
                <p className="text-gray-500">{node.type.replace('Node', '')}</p>
                {node.type === 'inputNode' && (
                  <p className="mt-1">Features: {node.data.features}</p>
                )}
                {node.type === 'linearNode' && (
                  <>
                    <p className="mt-1">Input features: {node.data.in_features}</p>
                    <p className="mt-1">Output features: {node.data.out_features}</p>
                  </>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-6">
          <h3 className="font-semibold mb-2">Shape Propagation</h3>
          <p className="text-sm text-gray-600">
            In neural networks, shape propagation is critical. For linear layers, the input features (in_features) 
            must match the number of features from the previous layer's output, and the output features (out_features) 
            define the dimensionality of this layer's output.
          </p>
        </div>

        <div className="mt-4 bg-yellow-50 border border-yellow-100 p-4 rounded-lg">
          <h3 className="font-semibold flex items-center">
            <AlertCircle className="h-4 w-4 mr-2 text-yellow-500" />
            Common Shape Mismatch Issue
          </h3>
          <p className="mt-2 text-sm">
            A common error is when the number of features in an input node's output doesn't match 
            the in_features parameter of the connected linear layer. Always ensure these values match exactly.
          </p>
          <pre className="mt-3 bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
            {`// INCORRECT:
input: { features: 784, outputShape: { features: 784 } }
linear: { in_features: 512, ... } // Mismatch: 784 ≠ 512

// CORRECT:
input: { features: 784, outputShape: { features: 784 } }
linear: { in_features: 784, ... } // Match: 784 = 784`}
          </pre>
        </div>
      </div>
    </div>
  );
}
