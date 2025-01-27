Class {
	#name : 'MeanSquaredError',
	#superclass : 'LossFunction',
	#category : 'AI-Neural-Network',
	#package : 'AI-Neural-Network'
}

{ #category : 'scoring' }
MeanSquaredError >> derivative: predictions targets: actuals [ 

    | n grad |
    n := predictions size.
    grad := Array new: n.
    1 to: n do: [:i |
        grad at: i put: ( (2.0 / n) * ((predictions at: i) - (actuals at: i)) ).
    ].
    ^ grad
]

{ #category : 'scoring' }
MeanSquaredError >> value: predictions targets: actuals [ 

    | sum n diff |
    sum := 0.0.
    n := predictions size.
    1 to: n do: [:i |
        diff := (predictions at: i) - (actuals at: i).
        sum := sum + (diff * diff).
    ].
    ^ sum / n
]
