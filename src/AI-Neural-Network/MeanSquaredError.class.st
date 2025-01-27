"
The MeanSquaredError class is a subclass of LossFunction that computes the Mean Squared Error (MSE) loss. It includes:
- value:targets:: Calculates the MSE loss between predictions and targets.
- derivative:targets:: Computes the gradient of the MSE loss for backpropagation.
"
Class {
	#name : 'MeanSquaredError',
	#superclass : 'LossFunction',
	#category : 'AI-Neural-Network-Loss Function',
	#package : 'AI-Neural-Network',
	#tag : 'Loss Function'
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
