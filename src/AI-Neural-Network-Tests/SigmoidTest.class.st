Class {
	#name : 'SigmoidTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
SigmoidTest >> testDerivative [

    | sigmoid x result expected sigVal |
    sigmoid := Sigmoid new.
    x := 0.0.
    sigVal := sigmoid value: x.
    result := sigmoid derivative: x. 
    expected := sigVal * (1.0 - sigVal).  
    self assert: (result closeTo: expected precision: 1e-6).
]

{ #category : 'tests' }
SigmoidTest >> testValue [

    | sigmoid x result expected |
    sigmoid := Sigmoid new.
    x := 0.0.
    result := sigmoid value: x.
    expected := 0.5.   "σ(0) = 0.5"
    self assert: (result closeTo: expected precision: 1e-6)

]
