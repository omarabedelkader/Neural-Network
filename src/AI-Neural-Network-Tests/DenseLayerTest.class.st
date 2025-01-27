Class {
	#name : 'DenseLayerTest',
	#superclass : 'TestCase',
	#instVars : [
		'layer',
		'mockActivation'
	],
	#category : 'AI-Neural-Network-Tests',
	#package : 'AI-Neural-Network-Tests'
}

{ #category : 'tests' }
DenseLayerTest >> setUp [

    layer := DenseLayer new.
    mockActivation := MockActivation new.
]

{ #category : 'tests' }
DenseLayerTest >> tearDown [

    layer := nil.
    mockActivation := nil.
]

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
DenseLayerTest >> testBackwardPass [

    "Check backward pass using a mock activation derivative = 1. 
    This simplifies dLoss/dZ = dLoss/dOut * derivative(Z) = dLoss/dOut * 1."
    | inputVector dLoss_dOut backwardResult dLoss_dZ dLoss_dIn |
    
    layer inputSize: 2 outputSize: 2 activation: mockActivation.
    layer weights: #(0.5 0.2   0.1 0.4) asFloatArray.
    layer biases: #(0.1 0.2) asFloatArray.

    "Forward pass first so that lastZ and lastInput are set."
    inputVector := #(1.0 2.0) asFloatArray.
    layer forward: inputVector.

    "Say the loss gradient wrt output is #(.1, .05).
     Then dLoss/dZ = dLoss/dOut * activation'(z) = #(.1, .05) * #(1, 1) = #(.1, .05)"
    dLoss_dOut := #(0.1 0.05) asFloatArray.
    backwardResult := layer backward: dLoss_dOut.

    dLoss_dZ := backwardResult first.  "should be #(0.1 0.05)"
    dLoss_dIn := backwardResult second.

    self assert: dLoss_dZ equals: #(0.1 0.05) asFloatArray.

    "Check dLoss_dIn:
       dLoss/dIn[i] = sum_j (weights[i,j]*dLoss_dZ[j]) 
       Flattened weight indices:
         w(1->1) = 0.5, w(1->2) = 0.2, w(2->1) = 0.1, w(2->2) = 0.4

       For input index 1:
         = 0.5*0.1 + 0.1*0.05 = 0.05 + 0.005 = 0.055
       For input index 2:
         = 0.2*0.1 + 0.4*0.05 = 0.02 + 0.02 = 0.04
    "
    self assert: dLoss_dIn = #(0.055 0.04) asFloatArray.
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
DenseLayerTest >> testForwardPass [
    "Use a small known set of weights, biases, and an identity-like activation to test forward pass."
    | inputVector output |
    layer inputSize: 2 outputSize: 2 activation: mockActivation.
    
    "Set weights and biases manually:
       Suppose weights = [ w11, w12,
                           w21, w22 ]
       Flattened as (w11 w12 w21 w22) (row-major style).
       So for out1: sum = w11*x1 + w12*x2 + b1
          for out2: sum = w21*x1 + w22*x2 + b2
    "
    layer weights: #(0.5 0.2   0.1 0.4) asFloatArray.
    layer biases: #(0.1 0.2) asFloatArray.

    "We define our mock activation to be identity, so forward output = Z."
    inputVector := #(1.0 2.0) asFloatArray.   
    output := layer forward: inputVector.

    "Compute manually:
       out1 = 0.5*1 + 0.2*2 + 0.1 = 0.5 + 0.4 + 0.1 = 1.0
       out2 = 0.1*1 + 0.4*2 + 0.2 = 0.1 + 0.8 + 0.2 = 1.1
    "
    self assert: output equals: #(1.0 1.1) asFloatArray.

    "Check internal storage for lastInput and lastZ"
    self assert: layer lastInput equals: #(1.0 2.0) asFloatArray.
    self assert: layer lastZ equals: #(1.0 1.1) asFloatArray.
    self assert: layer lastOutput equals: #(1.0 1.1) asFloatArray.
]

{ #category : 'tests' }
DenseLayerTest >> testInitialization [

    "Test default initialization of a newly created layer."
    | emptyFloatArray |
    layer initialize.
    emptyFloatArray := #() asFloatArray.

    self assert: layer weights equals: emptyFloatArray.
    self assert: layer biases equals: emptyFloatArray.
    self assert: layer lastInput equals: emptyFloatArray.
    self assert: layer lastZ equals: emptyFloatArray.
    self assert: layer lastOutput equals: emptyFloatArray.

    "Check that inputSize/outputSize are nil (or uninitialized)."
    self assert: layer inputSize isNil.
    self assert: layer outputSize isNil.
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
DenseLayerTest >> testInputSizeOutputSizeActivationInitialization [

    "Initialize with specific inputSize, outputSize, activation, 
    then verify weights/biases got randomized properly."
    layer inputSize: 2 outputSize: 3 activation: mockActivation.

    self assert: layer inputSize = 2.
    self assert: layer outputSize = 3.
    self assert: layer activation = mockActivation.

    "Check lengths of weights/biases."
    self assert: layer weights size = (2 * 3).
    self assert: layer biases size = 3.

    "Just check that weights/biases are FloatArrays."
    self assert: layer weights class equals: Array.
    self assert: layer biases class equals: Array.
]

{ #category : 'tests' }
DenseLayerTest >> testSettersAndGetters [

    "Check that setters/getters for inputSize, outputSize, activation, etc., work properly."
    layer inputSize: 5.
    self assert: layer inputSize equals: 5.

    layer outputSize: 3.
    self assert: layer outputSize equals: 3.

    layer activation: mockActivation.
    self assert: layer activation equals: mockActivation.

    layer weights: #(1 2 3 4 5 6) asFloatArray.
    self assert: layer weights equals: #(1 2 3 4 5 6) asFloatArray.

    layer biases: #(0.1 0.2 0.3) asFloatArray.
    self assert: layer biases equals: #(0.1 0.2 0.3) asFloatArray.
]

{ #category : 'tests' }
DenseLayerTest >> testUpdateWeights [
    "Check that updateWeights modifies weights and biases as expected."
    | inputVector dLoss_dZ oldWeights oldBiases learningRate |
    layer inputSize: 2 outputSize: 2 activation: mockActivation.
    layer weights: #(0.5 0.2   0.1 0.4) asFloatArray.
    layer biases: #(0.1 0.2) asFloatArray.

    "Forward pass sets lastInput."
    inputVector := #(1.0 2.0) asFloatArray.
    layer forward: inputVector.

    "Now we simulate the gradient of the loss wrt Z."
    dLoss_dZ := #(0.1 0.05) asFloatArray.

    oldWeights := layer weights copy.
    oldBiases := layer biases copy.

    learningRate := 0.5.
    layer updateWeights: dLoss_dZ learningRate: learningRate.

    "Check new weights:
       For W[j, i], 
         W'[j,i] = W[j,i] - lr * (lastInput[i] * dLoss_dZ[j]).
       Flattened indexing: 
         - output #1: wIndex = 1->(1,2) => #1->0.5, #2->0.2
         - output #2: wIndex = 3->0.1, #4->0.4
       If lastInput = #(1.0, 2.0) 
         new W[1,1] = 0.5 - 0.5*(1.0*0.1) = 0.5 - 0.05 = 0.45
         new W[1,2] = 0.2 - 0.5*(2.0*0.1) = 0.2 - 0.1  = 0.1
         new W[2,1] = 0.1 - 0.5*(1.0*0.05)= 0.1 - 0.025= 0.075
         new W[2,2] = 0.4 - 0.5*(2.0*0.05)= 0.4 - 0.05= 0.35

       For biases:
         b'[j] = b[j] - lr*(dLoss_dZ[j])
         new b1 = 0.1 - 0.5*0.1  = 0.1 - 0.05  = 0.05
         new b2 = 0.2 - 0.5*0.05 = 0.2 - 0.025 = 0.175
    "
    self assert: layer weights = #(0.45 0.1 0.075 0.35) asFloatArray description: 'Weights did not update as expected.'.
    self assert: layer biases = #(0.05 0.175) asFloatArray description: 'Biases did not update as expected.'.

]
