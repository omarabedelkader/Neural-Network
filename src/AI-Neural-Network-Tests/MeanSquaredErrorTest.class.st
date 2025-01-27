Class {
	#name : 'MeanSquaredErrorTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
MeanSquaredErrorTest >> testDerivativeTargets [

    | mse predictions targets result expected i n |
    mse := MeanSquaredError new.
    predictions := #(1.0 2.0 3.0) asArray.
    targets := #(1.0 1.0 2.0) asArray.

    "MSE derivative for each element = 2*(prediction - target)/n
     n = 3
     diffs = (0, 1, 1)
     result = (0, 2/3, 2/3)
    "
    n := predictions size.
    result := mse derivative: predictions targets: targets.

    1 to: n do: [:idx |
        (idx = 1)
            ifTrue: [ expected := 0 ]
            ifFalse: [ expected := (2.0 * ((predictions at: idx) - (targets at: idx))) / n ].
        self assert: ((result at: idx) closeTo: expected precision: 1e-6).
    ].
]

{ #category : 'tests' }
MeanSquaredErrorTest >> testValueTargets [

    | mse predictions targets result expected |
    mse := MeanSquaredError new.
    predictions := #(1.0 2.0 3.0) asArray.
    targets := #(1.0 1.0 2.0) asArray.
    "Manually compute MSE:
     diffs = (0, 1, 1)
     squares = (0, 1, 1)
     sum = 2
     average = 2 / 3 = 0.6666...
    "
    result := mse value: predictions targets: targets.
    expected := 2.0 / 3.0.
    self assert: (result closeTo: expected precision: 1e-6).
]
