Class {
	#name : 'LSTMLayer',
	#superclass : 'Object',
	#instVars : [
		'inputSize',
		'hiddenSize',
		'Wf',
		'Wi',
		'Wo',
		'Wc',
		'bf',
		'bi',
		'bo',
		'bc',
		'lastHidden',
		'lastCell',
		'fGate',
		'iGate',
		'oGate',
		'cTilde',
		'sigmoidActivation',
		'tanhActivation'
	],
	#category : 'AI-Neural-Network-Layers',
	#package : 'AI-Neural-Network',
	#tag : 'Layers'
}

{ #category : 'initialization' }
LSTMLayer >> combineInputs: xVector hidden: hVector [ 

    "Concatenate x and h for simpler matrix multiply:
      [x_1, x_2, ..., x_inSize, h_1, ..., h_hSize]
    "
    ^ (xVector copyWithAll: hVector)
]

{ #category : 'initialization' }
LSTMLayer >> forward: inputVector [ 

    "
    f_t = sigmoid(Wf * [x_t, h_{t-1}] + bf)
    i_t = sigmoid(Wi * [x_t, h_{t-1}] + bi)
    cTilde_t = tanh(Wc * [x_t, h_{t-1}] + bc)
    C_t = f_t * C_{t-1} + i_t * cTilde_t
    o_t = sigmoid(Wo * [x_t, h_{t-1}] + bo)
    h_t = o_t * tanh(C_t)
    "
    | combined inSize hSize combSize |

    inSize := inputSize.
    hSize := hiddenSize.
    "We combine input x_t and h_{t-1} for matrix multiplication convenience"
    combined := self combineInputs: inputVector hidden: lastHidden.
    combSize := inSize + hSize.

    "Compute gates"
    fGate := self gateForward: combined
                          weights: Wf
                           bias: bf
                     activation: sigmoidActivation.
    iGate := self gateForward: combined
                          weights: Wi
                           bias: bi
                     activation: sigmoidActivation.
    oGate := self gateForward: combined
                          weights: Wo
                           bias: bo
                     activation: sigmoidActivation.
    cTilde := self gateForward: combined
                            weights: Wc
                             bias: bc
                       activation: tanhActivation.

    "Update cell state"
    1 to: hSize do: [:i |
        | oldCell |
        oldCell := (lastCell at: i).
        lastCell at: i put: ((fGate at: i) * oldCell) + ((iGate at: i) * (cTilde at: i)).
    ].

    "Update hidden state"
    1 to: hSize do: [:i |
        lastHidden at: i put: ((oGate at: i) * (tanhActivation value: (lastCell at: i))).
    ].

    "We return the hidden state as the 'output' of this layer."
    ^ lastHidden
]

{ #category : 'initialization' }
LSTMLayer >> gateForward: combinedVector weights: w bias: b activation: anActivation [ 

    | hSize inSize totalSize gateOut offset sum |
    hSize := hiddenSize.
    totalSize := combinedVector size.  "should be inputSize + hiddenSize"
    gateOut := Array new: hSize withAll: 0.0.

    "We partition w into (Wx, Wh). For each gate neuron i in [1..hiddenSize]:"
    1 to: hSize do: [:i |
        sum := 0.0.
        1 to: totalSize do: [:j |
            | wIndex |
            wIndex := (i - 1) * totalSize + j.
            sum := sum + (w at: wIndex) * (combinedVector at: j).
        ].
        sum := sum + (b at: i).
        gateOut at: i put: (anActivation value: sum).
    ].
    ^ gateOut
]

{ #category : 'initialization' }
LSTMLayer >> initialize [ 

    super initialize.
    Wf := #() asFloatArray.
    Wi := #() asFloatArray.
    Wo := #() asFloatArray.
    Wc := #() asFloatArray.
    bf := #() asFloatArray.
    bi := #() asFloatArray.
    bo := #() asFloatArray.
    bc := #() asFloatArray.

    lastHidden := #() asFloatArray.
    lastCell := #() asFloatArray.

    fGate := #() asFloatArray.
    iGate := #() asFloatArray.
    oGate := #() asFloatArray.
    cTilde := #() asFloatArray.

    "Default activations"
    sigmoidActivation := Sigmoid new.
    tanhActivation := Tanh new.
]

{ #category : 'initialization' }
LSTMLayer >> inputSize: inSize hiddenSize: hidSize [ 

    | randScale sizeXF sizeHF |
    inputSize := inSize.
    hiddenSize := hidSize.


    randScale := 0.01.

    sizeXF := inputSize * hiddenSize. "for Wf, Wi, Wo, Wc"
    sizeHF := hiddenSize * hiddenSize.

    Wf := (1 to: (sizeXF + sizeHF)) collect: [:i | (Random new next - 0.5) * randScale ].
    Wi := (1 to: (sizeXF + sizeHF)) collect: [:i | (Random new next - 0.5) * randScale ].
    Wo := (1 to: (sizeXF + sizeHF)) collect: [:i | (Random new next - 0.5) * randScale ].
    Wc := (1 to: (sizeXF + sizeHF)) collect: [:i | (Random new next - 0.5) * randScale ].

    bf := (1 to: hiddenSize) collect: [:i | (Random new next - 0.5) * randScale ].
    bi := (1 to: hiddenSize) collect: [:i | (Random new next - 0.5) * randScale ].
    bo := (1 to: hiddenSize) collect: [:i | (Random new next - 0.5) * randScale ].
    bc := (1 to: hiddenSize) collect: [:i | (Random new next - 0.5) * randScale ].

    lastHidden := (1 to: hiddenSize) collect: [:i | 0.0 ].
    lastCell := (1 to: hiddenSize) collect: [:i | 0.0 ].

    ^ self
]
