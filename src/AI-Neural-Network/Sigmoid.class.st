Class {
	#name : 'Sigmoid',
	#superclass : 'Activation',
	#category : 'AI-Neural-Network',
	#package : 'AI-Neural-Network'
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
