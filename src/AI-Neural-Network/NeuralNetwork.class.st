"
The NeuralNetwork class manages the structure of a neural network and provides methods for training, predicting, and performing forward/backward propagation. Key instance variables and methods include:

- layers: An ordered collection of layers in the network.
- lossFunction: The loss function used for training.
- learningRate: Learning rate for weight updates.
- addLayer:: Adds a layer to the network.
- forward:: Performs a forward pass through the network.
- backward:targets:: Performs backpropagation to calculate gradients and update weights.
- trainOn:epochs:: Trains the network on a dataset for a specified number of epochs.
"
Class {
	#name : 'NeuralNetwork',
	#superclass : 'Object',
	#instVars : [
		'layers',
		'lossFunction',
		'learningRate'
	],
	#category : 'AI-Neural-Network-NN',
	#package : 'AI-Neural-Network',
	#tag : 'NN'
}

{ #category : 'initialization' }
NeuralNetwork >> addLayer: aLayer [ 

    layers add: aLayer.
]

{ #category : 'initialization' }
NeuralNetwork >> backward: predictions targets: actuals [ 
    "Perform backpropagation through the network and update weights"

    | gradientLossWithRespectToOutput gradientForCurrentLayer |
    gradientLossWithRespectToOutput := lossFunction derivative: predictions targets: actuals.
    
    "Propagate gradients from output layer to input layer"
    layers reverseDo: [:layer |
        | layerBackpropResult |
        layerBackpropResult := layer backward: gradientLossWithRespectToOutput.
        
        "layerBackpropResult contains gradients for current layer and input to previous layer"
        gradientForCurrentLayer := layerBackpropResult first.
        gradientLossWithRespectToOutput := layerBackpropResult second.

        "Update weights and biases for the current layer"
        layer 
            updateWeights: gradientForCurrentLayer 
            learningRate: learningRate.
    ].

    ^gradientLossWithRespectToOutput "Return final gradient for potential further use"


]

{ #category : 'initialization' }
NeuralNetwork >> forward: inputVector [ 

    | out |
    out := inputVector.
    layers do: [:layer |
        out := layer forward: out
    ].
    ^ out
]

{ #category : 'initialization' }
NeuralNetwork >> initialize [ 

    super initialize.
    layers := OrderedCollection new.
    lossFunction := MeanSquaredError new.
    learningRate := 0.1.  "default"
]

{ #category : 'initialization' }
NeuralNetwork >> predict: inputVector [ 

    | output |
    output := self forward: inputVector.
    ^ output
]

{ #category : 'initialization' }
NeuralNetwork >> setLearningRate: lr [ 

    learningRate := lr
]

{ #category : 'initialization' }
NeuralNetwork >> setLossFunction: aLossFunction [ 

    lossFunction := aLossFunction.
]

{ #category : 'initialization' }
NeuralNetwork >> trainOn: trainingData epochs: numEpochs [ 

    1 to: numEpochs do: [:epoch |
        trainingData do: [:assoc |
            | inputs targets predictions |
            inputs := assoc key asFloatArray.
            "Ensure target is also an array (for multi-output)."
            targets := assoc value isArray 
                        ifTrue: [ assoc value asFloatArray ] 
                        ifFalse: [ #( ) asFloatArray copyWith: assoc value ].

            "Forward"
            predictions := self forward: inputs.

            "Backward"
            self backward: predictions targets: targets.
        ].
    ].
]
