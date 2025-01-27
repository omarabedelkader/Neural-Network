"
a mock class
"
Class {
	#name : 'MockLayer',
	#superclass : 'Object',
	#instVars : [
		'lastForwardInput',
		'lastBackwardGradient',
		'weights',
		'biases'
	],
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'initialization' }
MockLayer >> backward: gradient [

    "A minimal 'backward' that just returns a fixed gradient shape for demonstration.
     We also store the incoming gradient for inspection."
    lastBackwardGradient := gradient.

    "Simulate returning two pieces:
     1) The gradient for the current layer (weights/bias updates).
     2) The gradient passed back to the previous layer."

    ^ { gradient. gradient } "Returning the same gradient to simulate pass-through"
]

{ #category : 'accessing' }
MockLayer >> biases [

	^ biases
]

{ #category : 'accessing' }
MockLayer >> biases: anObject [

	biases := anObject
]

{ #category : 'initialization' }
MockLayer >> initialize [

    weights := #(1 1 1).
    biases := #(0)
]

{ #category : 'accessing' }
MockLayer >> lastBackwardGradient [

	^ lastBackwardGradient
]

{ #category : 'accessing' }
MockLayer >> lastBackwardGradient: anObject [

	lastBackwardGradient := anObject
]

{ #category : 'accessing' }
MockLayer >> lastForwardInput [

	^ lastForwardInput
]

{ #category : 'accessing' }
MockLayer >> lastForwardInput: anObject [

	lastForwardInput := anObject
]

{ #category : 'initialization' }
MockLayer >> updateWeights: gradient learningRate: lr [

    "A trivial weight update: subtract gradient from each weight.
     This is purely for demonstration—real layers would handle actual
     matrix/array operations."
    weights := weights collect: [:w | w - (lr * (gradient first)) ].
    biases := biases collect: [:b | b - (lr * (gradient first)) ].
]

{ #category : 'accessing' }
MockLayer >> weights [

	^ weights
]

{ #category : 'accessing' }
MockLayer >> weights: anObject [

	weights := anObject
]
