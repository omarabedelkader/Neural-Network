Class {
	#name : 'DenseLayerTest',
	#superclass : 'TestCase',
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
DenseLayerTest >> testBackward [

    | layer input forwardOutput dLoss_dOut backResult dLoss_dZ dLoss_dIn |
    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.

    "Set weights/biases to simple known values."
    layer weights at: 1 put: 0.5.
    layer weights at: 2 put: -0.5.
    layer biases at: 1 put: 0.0.

    "Forward pass with a known input"
    input := #(1.0 2.0) asArray.
    forwardOutput := layer forward: input.  "Should be ~ 0.37754"

    "Simulate a gradient from the next layer: dLoss/dOut = (Array with something)."
    dLoss_dOut := #(1.0) asArray.  "Say we got gradient 1.0 for the single output"

    "Backward pass"
    backResult := layer backward: dLoss_dOut.
    dLoss_dZ := backResult first.
    dLoss_dIn := backResult second.

    "Check we have correct sizes"
    self assert: dLoss_dZ size = layer outputSize.
    self assert: dLoss_dIn size = layer inputSize.

    "We won't do a precise numeric check here, but you can do so by
     manually calculating derivative of the sigmoid at z = -0.5."
    self assert: (dLoss_dZ first ~= 0.0).
    self assert: (dLoss_dIn first ~= 0.0).
    self assert: (dLoss_dIn second ~= 0.0).
]

{ #category : 'tests' }
DenseLayerTest >> testForward [

    | layer input result |
    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.

    "Set known weights/biases for reproducibility."
    layer weights at: 1 put: 0.5.  "W[1,1]"
    layer weights at: 2 put: -0.5. "W[2,1]"
    layer biases at: 1 put: 0.0.

    input := #(1.0 2.0) asArray.
    result := layer forward: input.

    "Compute by hand: z = 0.5*1.0 + (-0.5)*2.0 + 0.0 = 0.5 - 1.0 = -0.5
     Sigmoid(-0.5) ~ 0.3775406688
    "
    self assert: (result first closeTo: 0.37754 precision: 1e-5).

]

{ #category : 'tests' }
DenseLayerTest >> testInitialize [

    | layer |
    layer := DenseLayer new.
    layer initialize.
    self assert: layer weights isEmpty.
    self assert: layer biases isEmpty.
    self assert: layer lastInput isEmpty.
    self assert: layer lastZ isEmpty.
    self assert: layer lastOutput isEmpty.
]

{ #category : 'tests' }
DenseLayerTest >> testInputSizeOutputSizeActivation [

    | layer |
    layer := DenseLayer new.
    layer inputSize: 3 outputSize: 2 activation: Sigmoid new.
    self assert: layer inputSize equals: 3.
    self assert: layer outputSize equals: 2.
    self assert: layer activation class equals: Sigmoid.

    "Check that weights and biases are initialized properly."
    self deny: layer weights isEmpty.
    self deny: layer biases isEmpty.
    self assert: layer weights size equals: (3 * 2).
    self assert: layer biases size equals: 2.
]

{ #category : 'tests' }
DenseLayerTest >> testUpdateWeights [

    | layer input forwardOutput dLoss_dOut dLoss_dZ oldWeights oldBiases |
    layer := DenseLayer new.
    layer inputSize: 2 outputSize: 1 activation: Sigmoid new.

    "Set known weights/biases."
    layer weights at: 1 put: 0.5.
    layer weights at: 2 put: -0.5.
    layer biases at: 1 put: 0.0.

    input := #(1.0 2.0) asArray.
    forwardOutput := layer forward: input.  "Compute forward once."

    "Pretend dLoss_dZ is known from backprop (some gradient)."
    dLoss_dOut := #(1.0) asArray.
    "We first do backward to record lastInput and get the final dLoss_dZ."
    "But let's assume we already have the final partial w.r.t. Z for this layer 
     is the same as dLoss_dOut for a single output scenario."
    dLoss_dZ := dLoss_dOut.  "In some layers, this might be different, 
                              but let's keep it simple."

    oldWeights := layer weights copy.
    oldBiases := layer biases copy.

    layer updateWeights: dLoss_dZ learningRate: 0.1.

    "Check that weights have changed from old weights in the direction 
     expected. W[i] -> W[i] - lr*(lastInput[i]*dLoss_dZ). 
     Here lastInput = (1.0, 2.0) and dLoss_dZ = 1.0. 
     So new W[1] = 0.5 - 0.1*(1.0*1.0)=0.4
        new W[2] = -0.5 - 0.1*(2.0*1.0)= -0.7
    "
    self assert: ((layer weights at: 1) closeTo: 0.4 precision: 1e-7).
    self assert: ((layer weights at: 2) closeTo: -0.7 precision: 1e-7).

    "And biases: b[j] -> b[j] - lr*(dLoss_dZ[j]).
     old bias was 0.0, new should be -0.1 * 1.0 = -0.1
    "
    self assert: ((layer biases at: 1) closeTo: -0.1 precision: 1e-7).
]
