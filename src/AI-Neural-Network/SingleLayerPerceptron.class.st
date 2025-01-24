Class {
	#name : 'SingleLayerPerceptron',
	#superclass : 'Object',
	#instVars : [
		'weights',
		'bias',
		'learningRate'
	],
	#category : 'AI-Neural-Network',
	#package : 'AI-Neural-Network'
}

{ #category : 'initialization' }
SingleLayerPerceptron >> activation: z [
    "Sigmoid activation."
    ^ 1.0 / (1.0 + (z negated exp)).
]

{ #category : 'initialization' }
SingleLayerPerceptron >> initialize [
    "Initialize with zero weights, zero bias, and a default learning rate."
    weights := #(0 0) asFloatArray.  "Example: 2 inputs -> array of length 2."
    bias := 0.0.
    learningRate := 0.1
]

{ #category : 'initialization' }
SingleLayerPerceptron >> predict: inputs [
    | sum |
    sum := 0.0.
    1 to: inputs size do: [ :i |
        sum := sum + ((inputs at: i) * (weights at: i)).
    ].
    sum := sum + bias.
    ^ self activation: sum
]

{ #category : 'initialization' }
SingleLayerPerceptron >> trainOn: inputs label: target [
    | prediction error |
    prediction := self predict: inputs.
    error := target - prediction.

    "Update weights"
    1 to: inputs size do: [ :i |
        weights 
            at: i 
            put: ((weights at: i) + (learningRate * error * (inputs at: i))).
    ].

    "Update bias"
    bias := bias + (learningRate * error).
]
