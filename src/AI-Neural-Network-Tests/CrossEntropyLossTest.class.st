Class {
	#name : 'CrossEntropyLossTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
CrossEntropyLossTest >> testBinaryCrossEntropyDerivative [
    "
    This test checks whether the CrossEntropyLoss >> derivative:targets:
    produces correct gradients for binary cross-entropy.
    
    derivative(i) = (p_i - y_i) / [p_i * (1 - p_i) + eps].
    "
    | predictions targets expectedGrad actualGrad lossFn delta |
    
    "Using the same two-sample example:
       Sample 1: p=0.9, y=1
         => p - y = -0.1
            p*(1-p) = 0.9*0.1 = 0.09
            derivative = -0.1 / 0.09 = -1.111111...
       Sample 2: p=0.2, y=0
         => p - y = 0.2
            p*(1-p) = 0.2*0.8 = 0.16
            derivative = 0.2 / 0.16 = 1.25
    "
    
    predictions := #(0.9 0.2) asFloatArray.
    targets := #(1 0) asFloatArray.
    expectedGrad := #(-1.111111 1.25) asFloatArray.  "approx."

    lossFn := CrossEntropyLoss new.
    actualGrad := lossFn derivative: predictions targets: targets.

    delta := 1e-5.

    1 to: actualGrad size do: [:i |
        self 
            assert: ((actualGrad at: i)
                closeTo: (expectedGrad at: i)
                precision delta).
    ].
]

{ #category : 'tests' }
CrossEntropyLossTest >> testBinaryCrossEntropyValue [
    "
    This test checks whether the CrossEntropyLoss >> value:targets:
    produces the expected binary cross-entropy for simple inputs.
    
    For a single data point in a binary classification context:
      Loss = -[ y*ln(p) + (1-y)*ln(1-p) ].
    We'll average over multiple points, since the code divides by n.
    "
    | predictions targets actualValue expectedValue lossFn delta n |
    
    "Example data (two samples): 
       1) p=0.9, y=1
       2) p=0.2, y=0

     Hand calculation:
       For i=1: 
         -[1*ln(0.9) + 0*ln(1-0.9)] = -ln(0.9) ~ 0.1053605
       For i=2:
         -[0*ln(0.2) + 1*ln(0.8)] = -ln(0.8) ~ 0.2231435
       Sum = 0.1053605 + 0.2231435 = 0.328504
       Average = 0.328504 / 2 = 0.164252
    "
    
    predictions := #(0.9 0.2) asFloatArray.
    targets := #(1 0) asFloatArray.
    expectedValue := 0.164252.  "Approx. from hand calculation."

    lossFn := CrossEntropyLoss new.
    actualValue := lossFn value: predictions targets: targets.
    
    "Check correctness within a small tolerance"
    delta := 1e-6.
    self assert: (actualValue ~= expectedValue).

    "Optional: You can also do a quick check it is > 0"
    self assert: actualValue > 0.

    "Also check an edge case: predictions = targets => near 0 (perfect classification)
     We'll test a single perfect match:
     y=1, p=1 => crossEntropy ~ -[1 * ln(1+eps)] => ~ 0"
    predictions := #(1.0) asFloatArray.
    targets := #(1) asFloatArray.
    actualValue := lossFn value: predictions targets: targets.
    self assert: actualValue < 1e-6
]
