"
Example for using this package

```smalltalk

| net trainingSamples epochs prediction |

""Create the network.""
net := NeuralNetwork new.
net setLearningRate: 0.1.  ""Set global LR.""

""Add layers. 
  Input size 2 -> hidden size 2 (for demonstration) -> activation Sigmoid.""
net addLayer: (DenseLayer new
                    inputSize: 2
                    outputSize: 2
                    activation: Sigmoid new).

""Second layer from hidden size 2 to output size 1 -> Sigmoid activation.""
net addLayer: (DenseLayer new
                    inputSize: 2
                    outputSize: 1
                    activation: Sigmoid new).

""Prepare training data (logical OR). 
 We wrap the labels in an array so it can handle multi-dimensional outputs.""
trainingSamples := {
    (#(0 0) -> #(0)).
    (#(0 1) -> #(1)).
    (#(1 0) -> #(1)).
    (#(1 1) -> #(1)).
}.

epochs := 2000000.

net trainOn: trainingSamples epochs: epochs.

""Test the network.""
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
"
Class {
	#name : 'ManifestAINeuralNetwork',
	#superclass : 'PackageManifest',
	#category : 'AI-Neural-Network-Manifest',
	#package : 'AI-Neural-Network',
	#tag : 'Manifest'
}
