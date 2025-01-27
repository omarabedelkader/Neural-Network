Class {
	#name : 'CrossEntropyLoss',
	#superclass : 'LossFunction',
	#category : 'AI-Neural-Network-Loss',
	#package : 'AI-Neural-Network',
	#tag : 'Loss'
}

{ #category : 'scoring' }
CrossEntropyLoss >> derivative: predictions targets: actuals [ 

    | grad n eps |
    n := predictions size.
    eps := 1e-12.
    grad := Array new: n.
    1 to: n do: [:i |
        grad at: i put: (
            ((predictions at: i) - (actuals at: i))
            /
            (((predictions at: i) * (1.0 - (predictions at: i))) + eps)
        ).
    ].
    ^ grad
]

{ #category : 'scoring' }
CrossEntropyLoss >> value: predictions targets: actuals [ 

    | sum n eps |
    eps := 1e-12.  "Prevent log(0)"
    sum := 0.0.
    n := predictions size.
    1 to: n do: [:i |
        sum := sum - ((actuals at: i) * ((predictions at: i) + eps log) +
                      (1 - (actuals at: i)) * (1 - (predictions at: i) + eps log)).
    ].
    ^ sum / n

]
