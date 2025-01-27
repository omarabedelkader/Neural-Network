Class {
	#name : 'LossFunctionTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
LossFunctionTest >> testAbstractMethods [

    | lf predictions targets |
    lf := LossFunction new.
    predictions := #(0.0 0.0) asArray.
    targets := #(1.0 1.0) asArray.

    self 
        should: [ lf value: predictions targets: targets ] 
        raise: SubclassResponsibility.

    self 
        should: [ lf derivative: predictions targets: targets ] 
        raise: SubclassResponsibility.
]
