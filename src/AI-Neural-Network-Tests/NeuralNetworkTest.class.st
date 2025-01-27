Class {
	#name : 'NeuralNetworkTest',
	#superclass : 'TestCase',
	#instVars : [
		'network',
		'mockLayer'
	],
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
NeuralNetworkTest >> setUp [

    network := NeuralNetwork new.
    mockLayer := MockLayer new.
]

{ #category : 'tests' }
NeuralNetworkTest >> tearDown [

    network := nil.
    mockLayer := nil.
]

{ #category : 'tests' }
NeuralNetworkTest >> testBackward [

    "We do a simple test with a single dense layer."
    | net layer input targets predictions finalGrad |
    net := NeuralNetwork new.
    net initialize.
    net setLearningRate: 0.1.

    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.
    net addLayer: layer.

    input := #(1.0 2.0) asArray.
    targets := #(0.0) asArray.

    "Forward pass"
    predictions := net forward: input. "some number"

    "Backward pass => update weights"
    finalGrad := net backward: predictions targets: targets.

    "finalGrad is the gradient w.r.t. the input of the entire net.
     We just check it's the right size. 
    "
    self assert: (finalGrad size = 2).
    "Also check that the layer's weights or biases changed a bit."
    "You could store old weights, do a forward, backward, compare."
]

{ #category : 'tests' }
NeuralNetworkTest >> testForward [

    | net layer input output |
    net := NeuralNetwork new.
    net initialize.

    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.
    net addLayer: layer.

    input := #(1.0 2.0) asArray.
    output := net forward: input.
    self assert: (output size = 1).
    "At least check it's between 0 and 1 for a sigmoid."
    self assert: ((output first >= 0.0) and: [output first <= 1.0]).
]

{ #category : 'tests' }
NeuralNetworkTest >> testSetLearningRate [
    "Test that setLearningRate: updates the instance variable."
    network setLearningRate: 0.05.
    self assert: network learningRate = 0.05.
]

{ #category : 'tests' }
NeuralNetworkTest >> testSetLossFunction [
    "Test that setLossFunction: updates the instance variable."
    | customLoss |
    customLoss := MeanSquaredError new.  "Or a different one, if you have multiple"
    network setLossFunction: customLoss.
    self assert: network lossFunction == customLoss.
]

{ #category : 'tests' }
NeuralNetworkTest >> testTrainOnEpochs [
    | net layer trainingData oldWeights oldBiases |
    net := NeuralNetwork new.
    net initialize.
    net setLearningRate: 0.1.

    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.
    net addLayer: layer.

    "We'll keep track of initial weights/biases"
    oldWeights := layer weights copy.
    oldBiases := layer biases copy.

    "Simulate training data with a couple of input-output pairs."
    trainingData := {
        ( #(1.0 2.0) -> #(0.0 ) ).
        ( #(0.0 0.0) -> #(1.0 ) )
    } asOrderedCollection.

    net trainOn: trainingData epochs: 2.

    "After training for 2 epochs, we expect changes in weights/biases."
    self deny: (layer weights = oldWeights).
    self deny: (layer biases = oldBiases).
]
