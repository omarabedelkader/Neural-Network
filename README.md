# Neural-Network

# AI-Neural-Network Package

This repository provides a basic implementation of a feedforward neural network. It contains classes for:
- Managing the overall network structure and training process (e.g., **NeuralNetwork**)
- Layers that implement forward and backward passes (e.g., **DenseLayer**)
- Activation functions (e.g., **ReLU**, **Sigmoid**, **Tanh**, **Identity**)
- Loss functions for training (e.g., **MeanSquaredError**, **CrossEntropyLoss**)

## Key Features
- **NeuralNetwork**: Creates a sequence of layers, handles forward propagation, backpropagation, and training.
- **DenseLayer**: Implements a fully-connected (dense) layer with optional activation functions.
- **Activation Functions**: Defines how each neuron's output is computed (ReLU, Sigmoid, Tanh, etc.).
- **Loss Functions**: Measures how far predictions are from the target outputs (MSE, CrossEntropy, etc.).

## Usage (High-Level)
1. Create an instance of **NeuralNetwork**.
2. Add one or more **DenseLayer** instances, specifying input/output sizes and activation functions.
3. Configure the **lossFunction** and **learningRate**.
4. Train the network on a dataset by calling `trainOn:epochs:`.
5. Use `predict:` for inference once the model is trained.

## Example:

```smalltalk

| net trainingSamples epochs prediction |

"Create the network."
net := NeuralNetwork new.
net setLearningRate: 0.1.  "Set global LR."

"Add layers. 
  Input size 2 -> hidden size 2 (for demonstration) -> activation Sigmoid."
net addLayer: (DenseLayer new
                    inputSize: 2
                    outputSize: 2
                    activation: Sigmoid new).

"Second layer from hidden size 2 to output size 1 -> Sigmoid activation."
net addLayer: (DenseLayer new
                    inputSize: 2
                    outputSize: 1
                    activation: Sigmoid new).

"Prepare training data (logical OR). 
 We wrap the labels in an array so it can handle multi-dimensional outputs."
trainingSamples := {
    (#(0 0) -> #(0)).
    (#(0 1) -> #(1)).
    (#(1 0) -> #(1)).
    (#(1 1) -> #(1)).
}.

epochs := 2000000.

net trainOn: trainingSamples epochs: epochs.

"Test the network."
Transcript cr; show: 'Predict (0 0): '.
prediction := net predict: #(0 0).
Transcript show: prediction printString.

Transcript cr; show: 'Predict (0 1): '.
prediction := net predict: #(0 1).
Transcript show: prediction printString.

Transcript cr; show: 'Predict (1 0): '.
prediction := net predict: #(1 0).
Transcript show: prediction printString.

Transcript cr; show: 'Predict (1 1): '.
prediction := net predict: #(1 1).
Transcript show: prediction printString.
```
