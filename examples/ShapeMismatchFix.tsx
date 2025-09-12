import React, { useState } from 'react';
import { Button } from "../components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Alert, AlertTitle, AlertDescription } from "../components/ui/alert";
import { Info, AlertCircle, Check, X } from "lucide-react";
import { useModelValidation } from "../lib/model-validator";

// Example with a shape mismatch error
const errorNodes = [
  {
    id: 'input-1',
    type: 'inputNode',
    position: { x: 100, y: 200 },
    data: {
      name: 'Input Layer',
      batch_size: 32,
      features: 784,  // Input features
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
      in_features: 512,  // ERROR: Doesn't match input features (784)
      out_features: 128,
      inputShape: {
        batch: 32,
        features: 784  // This matches the input, but in_features doesn't
      },
      outputShape: {
        batch: 32,
        features: 128
      }
    }
  }
];

// Example with fixed shape matching
const fixedNodes = [
  {
    id: 'input-1',
    type: 'inputNode',
    position: { x: 100, y: 200 },
    data: {
      name: 'Input Layer',
      batch_size: 32,
      features: 784,  // Input features
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
      in_features: 784,  // FIXED: Now matches input features (784)
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
  }
];

const edges = [
  {
    id: 'edge-1',
    source: 'input-1',
    target: 'linear-1',
    type: 'default'
  }
];

export default function ShapeMismatchFix() {
  const { validateModel } = useModelValidation();
  const [errorValidation, setErrorValidation] = useState(null);
  const [fixedValidation, setFixedValidation] = useState(null);

  const validateErrorModel = () => {
    const result = validateModel(errorNodes, edges);
    setErrorValidation(result);
  };

  const validateFixedModel = () => {
    const result = validateModel(fixedNodes, edges);
    setFixedValidation(result);
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Fixing Shape Mismatches</h1>

      <Tabs defaultValue="error" className="mb-6">
        <TabsList>
          <TabsTrigger value="error" className="flex items-center">
            <X className="h-4 w-4 mr-2 text-red-500" />
            With Error
          </TabsTrigger>
          <TabsTrigger value="fixed" className="flex items-center">
            <Check className="h-4 w-4 mr-2 text-green-500" />
            Fixed Version
          </TabsTrigger>
        </TabsList>

        <TabsContent value="error" className="mt-4">
          <div className="mb-4 flex justify-end">
            <Button onClick={validateErrorModel}>Validate Model</Button>
          </div>

          {errorValidation && (
            <Alert className={errorValidation.isValid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}>
              <AlertTitle className={`flex items-center ${errorValidation.isValid ? "text-green-800" : "text-red-800"}`}>
                {errorValidation.isValid ? 
                  <Check className="h-4 w-4 mr-2" /> : 
                  <AlertCircle className="h-4 w-4 mr-2" />
                }
                {errorValidation.isValid ? "Model Validation Successful" : "Model Validation Failed"}
              </AlertTitle>
              <AlertDescription>
                {!errorValidation.isValid && (
                  <div className="text-red-700">
                    <h3 className="font-semibold mt-2">Errors ({errorValidation.errors.length})</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {errorValidation.errors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}

          <div className="mt-6 bg-white border rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-3">Problem: Shape Mismatch</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border rounded-lg p-4 bg-gray-50">
                <h3 className="font-medium text-primary">Input Node</h3>
                <div className="mt-2 space-y-1 text-sm">
                  <p><span className="font-medium">Features:</span> 784</p>
                  <p><span className="font-medium">Output Shape:</span> {`{ batch: 32, features: 784 }`}</p>
                </div>
              </div>

              <div className="border rounded-lg p-4 bg-gray-50">
                <h3 className="font-medium text-primary">Linear Node</h3>
                <div className="mt-2 space-y-1 text-sm">
                  <p className="font-medium text-red-500">Problem:</p>
                  <p><span className="font-medium">in_features:</span> 512 <span className="text-red-500">(≠ 784)</span></p>
                  <p><span className="font-medium">Input Shape:</span> {`{ batch: 32, features: 784 }`}</p>
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm text-gray-600">
              <p>The problem is that the <code className="text-red-500 bg-red-50 px-1 rounded">in_features</code> parameter of the linear layer is 512, but it receives input with 784 features from the previous layer.</p>
            </div>

            <pre className="mt-3 bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
              {JSON.stringify(errorNodes, null, 2)}
            </pre>
          </div>
        </TabsContent>

        <TabsContent value="fixed" className="mt-4">
          <div className="mb-4 flex justify-end">
            <Button onClick={validateFixedModel}>Validate Model</Button>
          </div>

          {fixedValidation && (
            <Alert className={fixedValidation.isValid ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}>
              <AlertTitle className={`flex items-center ${fixedValidation.isValid ? "text-green-800" : "text-red-800"}`}>
                {fixedValidation.isValid ? 
                  <Check className="h-4 w-4 mr-2" /> : 
                  <AlertCircle className="h-4 w-4 mr-2" />
                }
                {fixedValidation.isValid ? "Model Validation Successful" : "Model Validation Failed"}
              </AlertTitle>
              <AlertDescription>
                {fixedValidation.isValid ? (
                  <p className="text-green-700">Your model has no errors and is ready to be used.</p>
                ) : (
                  <div className="text-red-700">
                    <h3 className="font-semibold mt-2">Errors ({fixedValidation.errors.length})</h3>
                    <ul className="list-disc pl-5 space-y-1">
                      {fixedValidation.errors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </AlertDescription>
            </Alert>
          )}

          <div className="mt-6 bg-white border rounded-lg p-4">
            <h2 className="text-lg font-semibold mb-3">Solution: Matching Shapes</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border rounded-lg p-4 bg-gray-50">
                <h3 className="font-medium text-primary">Input Node</h3>
                <div className="mt-2 space-y-1 text-sm">
                  <p><span className="font-medium">Features:</span> 784</p>
                  <p><span className="font-medium">Output Shape:</span> {`{ batch: 32, features: 784 }`}</p>
                </div>
              </div>

              <div className="border rounded-lg p-4 bg-gray-50">
                <h3 className="font-medium text-primary">Linear Node</h3>
                <div className="mt-2 space-y-1 text-sm">
                  <p className="font-medium text-green-500">Fixed:</p>
                  <p><span className="font-medium">in_features:</span> 784 <span className="text-green-500">(= 784)</span></p>
                  <p><span className="font-medium">Input Shape:</span> {`{ batch: 32, features: 784 }`}</p>
                </div>
              </div>
            </div>

            <div className="mt-4 text-sm text-gray-600">
              <p>The solution is to ensure that the <code className="text-green-500 bg-green-50 px-1 rounded">in_features</code> parameter of the linear layer matches the number of features from the previous layer's output.</p>
            </div>

            <pre className="mt-3 bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
              {JSON.stringify(fixedNodes, null, 2)}
            </pre>
          </div>

          <div className="mt-6 bg-blue-50 border border-blue-100 p-4 rounded-lg">
            <h3 className="font-semibold flex items-center">
              <Info className="h-4 w-4 mr-2 text-blue-500" />
              Key Insight
            </h3>
            <p className="mt-2 text-sm">
              In PyTorch and other deep learning frameworks, <strong>tensors have fixed shapes</strong>. When connecting layers, 
              the output dimensions of one layer must match the expected input dimensions of the next layer. For linear layers, 
              this means the <code className="bg-gray-100 px-1 rounded">in_features</code> parameter must match the number of 
              features from the previous layer's output.
            </p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
