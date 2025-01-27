"
The Tanh class implements the hyperbolic tangent (tanh) activation function.

"
Class {
	#name : 'Tanh',
	#superclass : 'Activation',
	#category : 'AI-Neural-Network-Activation',
	#package : 'AI-Neural-Network',
	#tag : 'Activation'
}

{ #category : 'accessing' }
Tanh >> derivative: x [
    "
    derivative of tanh(x) = 1 - tanh(x)^2
    "
    | y |
    y := self value: x.
    ^ 1.0 - (y * y)
]

{ #category : 'accessing' }
Tanh >> value: x [
    "
    Return tanh(x). We can compute this by:
       tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    or use another stable expression:
       tanh(x) = 2 / (1 + exp(-2x)) - 1
    "
    ^ ((2.0 / (1.0 + ((2.0 * x) negated exp))) - 1.0)
]
