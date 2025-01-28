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

{ #category : 'metrics' }
NeuralNetwork >> accuracy: predictions targets: actuals [
    "Compute accuracy = (TP + TN) / (TP + TN + FP + FN)."
    
    | cm tp tn fp fn total correct |
    cm := self confusionMatrix: predictions targets: actuals.
    tp := cm at: #TP.
    tn := cm at: #TN.
    fp := cm at: #FP.
    fn := cm at: #FN.

    total := tp + tn + fp + fn.
    total = 0 ifTrue: [ ^ 0.0 ].

    correct := tp + tn.
    ^ correct asFloat / total asFloat.
]

{ #category : 'initialization' }
NeuralNetwork >> addLayer: aLayer [ 

    layers add: aLayer.
]

{ #category : 'backward' }
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

{ #category : 'metrics' }
NeuralNetwork >> confusionMatrix: predictions targets: actuals [

    "Compute the confusion matrix for binary classification.
     predictions: an Array of floating-point predictions, e.g. in [0,1].
     actuals:     an Array of ground truth labels, 0 or 1.

     Returns a Dictionary with keys #TP #TN #FP #FN."
    
    | tp tn fp fn threshold predLabel actualLabel n |
    tp := 0.
    tn := 0.
    fp := 0.
    fn := 0.
    threshold := 0.5.
    n := predictions size.

    1 to: n do: [:i |
        "Convert continuous prediction to binary."
        predLabel := ((predictions at: i) >= threshold) 
                        ifTrue: [1] 
                        ifFalse: [0].
        actualLabel := (actuals at: i).

        (predLabel = 1 and: [actualLabel = 1]) ifTrue: [ tp := tp + 1 ].
        (predLabel = 0 and: [actualLabel = 0]) ifTrue: [ tn := tn + 1 ].
        (predLabel = 1 and: [actualLabel = 0]) ifTrue: [ fp := fp + 1 ].
        (predLabel = 0 and: [actualLabel = 1]) ifTrue: [ fn := fn + 1 ].
    ].

    ^ Dictionary newFromPairs: {
        #TP -> tp.
        #TN -> tn.
        #FP -> fp.
        #FN -> fn.
    }.
]

{ #category : 'metrics' }
NeuralNetwork >> f1Score: predictions targets: actuals [

    "Compute F1 = 2 * precision * recall / (precision + recall)."
    | prec rec denom |
    prec := self precision: predictions targets: actuals.
    rec := self recall: predictions targets: actuals.
    denom := prec + rec.
    denom = 0.0 ifTrue: [ ^ 0.0 ].
    ^ (2.0 * prec * rec) / denom
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

{ #category : 'accessing' }
NeuralNetwork >> layers [

	^ layers
]

{ #category : 'accessing' }
NeuralNetwork >> layers: anObject [

	layers := anObject
]

{ #category : 'accessing' }
NeuralNetwork >> learningRate [

	^ learningRate
]

{ #category : 'accessing' }
NeuralNetwork >> lossFunction [

	^ lossFunction
]

{ #category : 'metrics' }
NeuralNetwork >> precision: predictions targets: actuals [
    "Compute precision = TP / (TP + FP)."
    
    | cm tp fp denom |
    cm := self confusionMatrix: predictions targets: actuals.
    tp := cm at: #TP.
    fp := cm at: #FP.
    denom := tp + fp.
    denom = 0 ifTrue: [ ^ 0.0 ].
    ^ tp asFloat / denom asFloat.

]

{ #category : 'prediction' }
NeuralNetwork >> predict: inputVector [ 

    | output |
    output := self forward: inputVector.
    ^ output
]

{ #category : 'metrics' }
NeuralNetwork >> recall: predictions targets: actuals [

    "Compute recall = TP / (TP + FN)."
    | cm tp fn denom |
    cm := self confusionMatrix: predictions targets: actuals.
    tp := cm at: #TP.
    fn := cm at: #FN.
    denom := tp + fn.
    denom = 0 ifTrue: [ ^ 0.0 ].
    ^ tp asFloat / denom asFloat.
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
