Class {
	#name : 'NeuralNetworkTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
NeuralNetworkTest >> testAddLayer [

    | net layer |
    net := NeuralNetwork new.
    net initialize.
    layer := DenseLayer new.
    net addLayer: layer.
    self assert: (net layers size = 1).
    self assert: (net layers includes: layer).

]

{ #category : 'tests' }
NeuralNetworkTest >> testBackwardTargets [

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
NeuralNetworkTest >> testInitialize [

    | net |
    net := NeuralNetwork new.
    net initialize.
    self assert: net layers isEmpty.
    self assert: net lossFunction isKindOf: LossFunction.
    self assert: net learningRate = 0.1.
]

{ #category : 'tests' }
NeuralNetworkTest >> testSetLearningRate [

    | net |
    net := NeuralNetwork new.
    net initialize.
    net setLearningRate: 0.01.
    self assert: net learningRate = 0.01.

]

{ #category : 'tests' }
NeuralNetworkTest >> testSetLossFunction [

    | net mse |
    net := NeuralNetwork new.
    net initialize.
    mse := MeanSquaredError new.
    net setLossFunction: mse.
    self assert: net lossFunction == mse.
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
