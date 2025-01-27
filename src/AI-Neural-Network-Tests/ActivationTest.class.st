Class {
	#name : 'ActivationTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
ActivationTest >> testAbstractMethods [

    "We expect #value: and #derivative: to raise SubclassResponsibility
     if called directly on Activation."
    self 
        should: [ Activation new value: 1.0 ] 
        raise: SubclassResponsibility.

    self 
        should: [ Activation new derivative: 1.0 ] 
        raise: SubclassResponsibility.
]
