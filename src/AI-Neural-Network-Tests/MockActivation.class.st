"
a mock class
"
Class {
	#name : 'MockActivation',
	#superclass : 'Object',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'accessing' }
MockActivation >> derivative: z [

    "Constant derivative: f'(z) = 1"
    ^ 1
]

{ #category : 'accessing' }
MockActivation >> value: z [

    "Identity: f(z) = z"
    ^ z
]
