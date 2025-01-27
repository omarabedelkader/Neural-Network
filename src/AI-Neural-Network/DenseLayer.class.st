"
The DenseLayer class represents a fully connected neural network layer. It manages the weights, biases, and forward/backward operations for this layer. Key instance variables include:
- inputSize: The number of input nodes.
- outputSize: The number of output nodes.
- weights: Weight matrix connecting inputs to outputs.
- biases: Bias vector for each output.
- activation: Activation function applied to the layer's outputs.
- lastInput, lastZ, lastOutput: Intermediate values stored during forward propagation for use in backpropagation.
"
Class {
	#name : 'DenseLayer',
	#superclass : 'Object',
	#instVars : [
		'inputSize',
		'outputSize',
		'weights',
		'biases',
		'activation',
		'lastInput',
		'lastZ',
		'lastOutput'
	],
	#category : 'AI-Neural-Network',
	#package : 'AI-Neural-Network'
}

{ #category : 'accessing' }
DenseLayer >> activation [

	^ activation
]

{ #category : 'accessing' }
DenseLayer >> activation: anObject [

	activation := anObject
]

{ #category : 'initialization' }
DenseLayer >> backward: dLoss_dOut [ 

	"Return dLoss_dZ for this layer and dLoss_dIn for the previous layer"
   | dLoss_dZ dLoss_dIn |
   dLoss_dZ := Array new: outputSize.
   dLoss_dIn := Array new: inputSize.

   "dLoss/dZ = dLoss/dOut * derivative(activation, Z)."
   1 to: outputSize do: [:j |
       dLoss_dZ at: j put: ((dLoss_dOut at: j) * 
                             (activation derivative: (lastZ at: j))).
   ].

   1 to: inputSize do: [:i |
       | sum |
       sum := 0.0.
       1 to: outputSize do: [:j |
           | wIndex |
           wIndex := (j - 1) * inputSize + i.
           sum := sum + (weights at: wIndex) * (dLoss_dZ at: j).
       ].
       dLoss_dIn at: i put: sum.
    ].

    ^ { dLoss_dZ. dLoss_dIn }  
]

{ #category : 'accessing' }
DenseLayer >> biases [

	^ biases
]

{ #category : 'accessing' }
DenseLayer >> biases: anObject [

	biases := anObject
]

{ #category : 'initialization' }
DenseLayer >> forward: inputVector [ 

    lastInput := inputVector copy.
    lastZ := Array new: outputSize withAll: 0.0.
    
    1 to: outputSize do: [:outIdx |
        | sum |
        sum := 0.0.
        1 to: inputSize do: [:inIdx |
            | wIndex |
            wIndex := (outIdx - 1) * inputSize + inIdx. "Flattened 2D index."
            sum := sum + (weights at: wIndex) * (inputVector at: inIdx).
        ].
        sum := sum + (biases at: outIdx).
        lastZ at: outIdx put: sum.
    ].

    lastOutput := Array new: outputSize withAll: 0.0.
    1 to: outputSize do: [:i |
        lastOutput at: i put: (activation value: (lastZ at: i)).
    ].
    ^ lastOutput

]

{ #category : 'initialization' }
DenseLayer >> initialize [ 

    super initialize.
    weights := #() asFloatArray. 
    biases := #() asFloatArray.
    lastInput := #() asFloatArray.
    lastZ := #() asFloatArray.
    lastOutput := #() asFloatArray.


]

{ #category : 'accessing' }
DenseLayer >> inputSize [

	^ inputSize
]

{ #category : 'accessing' }
DenseLayer >> inputSize: anObject [

	inputSize := anObject
]

{ #category : 'initialization' }
DenseLayer >> inputSize: anInteger outputSize: anotherInteger activation: anActivation [ 

   | randomScale |   
	inputSize := anInteger.
   outputSize := anotherInteger.
  	activation := anActivation.
   randomScale := 0.01.

	weights := ((1 to: inputSize * outputSize) collect: [:k |
    	(Random new next - 0.5) * randomScale
	]) asArray asFloatArray.

   biases := (1 to: outputSize) collect: [:k |
       (Random new next - 0.5) * randomScale
    ].

 ^ self
]

{ #category : 'accessing' }
DenseLayer >> lastInput [

	^ lastInput
]

{ #category : 'accessing' }
DenseLayer >> lastInput: anObject [

	lastInput := anObject
]

{ #category : 'accessing' }
DenseLayer >> lastOutput [

	^ lastOutput
]

{ #category : 'accessing' }
DenseLayer >> lastOutput: anObject [

	lastOutput := anObject
]

{ #category : 'accessing' }
DenseLayer >> lastZ [

	^ lastZ
]

{ #category : 'accessing' }
DenseLayer >> lastZ: anObject [

	lastZ := anObject
]

{ #category : 'accessing' }
DenseLayer >> outputSize [

	^ outputSize
]

{ #category : 'accessing' }
DenseLayer >> outputSize: anObject [

	outputSize := anObject
]

{ #category : 'initialization' }
DenseLayer >> updateWeights: dLoss_dZ learningRate: learningRate [ 

    "Update weights and biases using the gradients from backpropagation."
    | gradientLossWithRespectToOutput |
    gradientLossWithRespectToOutput := dLoss_dZ.

    "Update weights: W[i, j] -= learningRate * dW[i, j]"
    1 to: outputSize do: [:j |
        1 to: inputSize do: [:i |
            | wIndex dW |
            wIndex := (j - 1) * inputSize + i. "Flattened 2D index"
            dW := (lastInput at: i) * (gradientLossWithRespectToOutput at: j).
            weights at: wIndex put: ((weights at: wIndex) - (learningRate * dW)).
        ].
    ].

    "Update biases: b[j] -= learningRate * db[j]"
    1 to: outputSize do: [:j |
        biases at: j put: ((biases at: j) - (learningRate * (gradientLossWithRespectToOutput at: j))).
    ].

 	^self 


]

{ #category : 'accessing' }
DenseLayer >> weights [

	^ weights
]

{ #category : 'accessing' }
DenseLayer >> weights: anObject [

	weights := anObject
]
