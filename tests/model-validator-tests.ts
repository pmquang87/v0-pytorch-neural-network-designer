import { ModelValidator } from '../lib/model-validator';

// Test function to verify shape compatibility validation
function testShapeCompatibility() {
  const validator = new ModelValidator();

  // Test case 1: Basic matching shapes
  const shape1 = { batch: 32, features: 784 };
  const shape2 = { batch: 32, features: 784 };
  console.assert(validator.testShapeCompatibility(shape1, shape2), 'Test 1 failed: Basic matching shapes');

  // Test case 2: Different feature dimensions should fail
  const shape3 = { batch: 32, features: 784 };
  const shape4 = { batch: 32, features: 512 };
  console.assert(!validator.testShapeCompatibility(shape3, shape4), 'Test 2 failed: Different features should not be compatible');

  // Test case 3: Dynamic dimensions should be compatible with any value
  const shape5 = { batch: 'dynamic', features: 784 };
  const shape6 = { batch: 32, features: 784 };
  console.assert(validator.testShapeCompatibility(shape5, shape6), 'Test 3 failed: Dynamic dimensions should be compatible');

  // Test case 4: Linear layer with in_features matching input features
  const shape7 = { batch: 32, features: 784 };
  const shape8 = { batch: 32, in_features: 784, out_features: 128 };
  console.assert(validator.testShapeCompatibility(shape7, shape8), 'Test 4 failed: Linear layer with matching in_features');

  // Test case 5: Linear layer with mismatched in_features
  const shape9 = { batch: 32, features: 784 };
  const shape10 = { batch: 32, in_features: 512, out_features: 128 };
  console.assert(!validator.testShapeCompatibility(shape9, shape10), 'Test 5 failed: Linear layer with mismatched in_features');

  // Test case 6: Convolutional layer shape validation
  const shape11 = { batch: 32, channels: 3, height: 224, width: 224 };
  const shape12 = { batch: 32, in_channels: 3, out_channels: 64, kernel_size: 3 };
  console.assert(validator.testShapeCompatibility(shape11, shape12), 'Test 6 failed: Conv layer with matching in_channels');

  // Test case 7: Convolutional layer with mismatched channels
  const shape13 = { batch: 32, channels: 3, height: 224, width: 224 };
  const shape14 = { batch: 32, in_channels: 1, out_channels: 64, kernel_size: 3 };
  console.assert(!validator.testShapeCompatibility(shape13, shape14), 'Test 7 failed: Conv layer with mismatched in_channels');

  console.log('Shape compatibility tests completed');
}

// Test function to verify model validation
function testModelValidation() {
  const validator = new ModelValidator();

  // Test case 1: Simple valid model
  const validNodes = [
    {
      id: 'input-1',
      type: 'inputNode',
      data: {
        batch_size: 32,
        features: 784,
        outputShape: { batch: 32, features: 784 }
      }
    },
    {
      id: 'linear-1',
      type: 'linearNode',
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

  const validResult = validator.validateModel(validNodes, validEdges);
  console.assert(validResult.isValid, 'Valid model test failed');
  console.assert(validResult.errors.length === 0, 'Valid model should have no errors');

  // Test case 2: Model with shape mismatch
  const invalidNodes = [
    {
      id: 'input-1',
      type: 'inputNode',
      data: {
        batch_size: 32,
        features: 784,
        outputShape: { batch: 32, features: 784 }
      }
    },
    {
      id: 'linear-1',
      type: 'linearNode',
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

  const invalidResult = validator.validateModel(invalidNodes, invalidEdges);
  console.assert(!invalidResult.isValid, 'Invalid model test failed');
  console.assert(invalidResult.errors.length > 0, 'Invalid model should have errors');
  console.assert(
    invalidResult.errors.some(err => err.includes('mismatch')),
    'Invalid model should report shape mismatch'
  );

  console.log('Model validation tests completed');
}

// Run the tests
export function runModelValidatorTests() {
  console.log('Running model validator tests...');
  testShapeCompatibility();
  testModelValidation();
  console.log('All tests completed');
}

// Uncomment to run tests
// runModelValidatorTests();
