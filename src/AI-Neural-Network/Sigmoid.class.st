"
The Sigmoid class is a subclass of Activation that implements the sigmoid activation function. It provides:
- value: Computes the sigmoid value for a given input.
- derivative: Computes the derivative of the sigmoid function for a given input.
"
Class {
	#name : 'Sigmoid',
	#superclass : 'Activation',
	#category : 'AI-Neural-Network-Activation',
	#package : 'AI-Neural-Network',
	#tag : 'Activation'
}

{ #category : 'accessing' }
Sigmoid >> derivative: x [ 

    | s |
    s := self value: x.
    ^ s * (1.0 - s).
]

{ #category : 'accessing' }
Sigmoid >> value: x [ 

    ^ 1.0 / (1.0 + (x negated exp))

]
