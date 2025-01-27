"
The ReLU class implements the Rectified Linear Unit activation function.

"
Class {
	#name : 'ReLU',
	#superclass : 'Activation',
	#category : 'AI-Neural-Network-Activation',
	#package : 'AI-Neural-Network',
	#tag : 'Activation'
}

{ #category : 'accessing' }
ReLU >> derivative: x [
    "
    derivative of ReLU(x) = 1 if x > 0, else 0
    "
    ^ x > 0.0
        ifTrue: [ 1.0 ]
        ifFalse: [ 0.0 ]
]

{ #category : 'accessing' }
ReLU >> value: x [
    "
    ReLU(x) = max(0, x)
    "
    ^ x max: 0.0
]
