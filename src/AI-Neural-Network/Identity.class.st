Class {
	#name : 'Identity',
	#superclass : 'Activation',
	#category : 'AI-Neural-Network-Activation',
	#package : 'AI-Neural-Network',
	#tag : 'Activation'
}

{ #category : 'accessing' }
Identity >> derivative: x [ 

    "d/dx (x) = 1"
    ^ 1.0

]

{ #category : 'accessing' }
Identity >> value: x [ 

    "f(x) = x"
    ^ x
]
