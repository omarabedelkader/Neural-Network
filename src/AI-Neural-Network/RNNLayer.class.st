Class {
	#name : 'RNNLayer',
	#superclass : 'Object',
	#instVars : [
		'inputSize',
		'hiddenSize',
		'outputSize',
		'Wxh',
		'Whh',
		'Why',
		'bh',
		'by',
		'activation',
		'lastInput',
		'lastHidden',
		'lastOutput'
	],
	#category : 'AI-Neural-Network-Layers',
	#package : 'AI-Neural-Network',
	#tag : 'Layers'
}

{ #category : 'initialization' }
RNNLayer >> backward: dLoss_dOut [ 

    "Compute gradients wrt hidden state and input via chain rule.
     dLoss_dOut is gradient from next layer (or final loss) wrt the output y_t.
     Return { dLoss_dHidden, dLoss_dIn } so that the previous layer can continue backprop.
    "

    | dLoss_dHidden dLoss_dIn dLoss_dWhy dLoss_dBy hSize inSize oSize |
    
    inSize := inputSize.
    hSize := hiddenSize.
    oSize := outputSize.

    dLoss_dHidden := Array new: hSize withAll: 0.0.
    dLoss_dIn := Array new: inSize withAll: 0.0.

    "1) Grad wrt Why and by, plus accumulate gradient to hidden"
    1 to: oSize do: [:outIdx |
        | gradOutput |
        gradOutput := (dLoss_dOut at: outIdx).  "the chain gradient from outside"

        "accumulate to each hidden unit"
        1 to: hSize do: [:hIdx |
            | wIndex |
            wIndex := (outIdx - 1) * hSize + hIdx.
            "dLoss/dWhy[outIdx,hIdx] = h(t) * gradOutput"
            "We do the actual weight update outside or store it here."

            "Now propagate to hidden"
            dLoss_dHidden at: hIdx put:
                ((dLoss_dHidden at: hIdx) + (Why at: wIndex) * gradOutput).
        ].
        "Also gradient wrt by"
        " dLoss/dby[outIdx] = gradOutput "
    ].

    "2) Now handle hidden -> hidden backprop. hidden is activation( ... ) so we multiply by derivative"
    1 to: hSize do: [:hIdx |
        | derivativeVal oldValue |
        " derivativeVal = derivative of activation at lastHidden[hIdx]"
        oldValue := (lastHidden at: hIdx). 
        derivativeVal := activation derivative: oldValue.

        dLoss_dHidden at: hIdx put: ((dLoss_dHidden at: hIdx) * derivativeVal).
    ].

    "3) Now propagate to input and previous hidden"
    1 to: hSize do: [:hIdx |
        | hiddenGrad |
        hiddenGrad := (dLoss_dHidden at: hIdx).

        " contributions to dLoss/dInput"
        1 to: inSize do: [:i |
            | wIndex |
            wIndex := (hIdx - 1) * inSize + i.
            dLoss_dIn at: i put: ((dLoss_dIn at: i) + (Wxh at: wIndex) * hiddenGrad).
        ].
        " contributions to dLoss/dHidden_{t-1}"
        " you might pass this upstream if you are unrolling over time."
    ].

    "Return pair: gradient wrt current hidden (for next step's backprop through time) and input."
    ^ { dLoss_dHidden. dLoss_dIn }
]

{ #category : 'initialization' }
RNNLayer >> forward: inputVector [ 

    "Compute h_t = activation(Wxh x_t + Whh h_{t-1} + b_h)
     Then y_t = Why h_t + b_y (if outputSize > 0)"

    | hiddenSum outputSum hSize oSize inSize hVec oVec |

    inSize := inputSize.
    hSize := hiddenSize.
    oSize := outputSize.

    lastInput := inputVector copy. "Store for backprop"
    
    "Compute hidden state"
    hiddenSum := Array new: hSize withAll: 0.0.
    1 to: hSize do: [:hIdx |
        | sum |
        sum := 0.0.
        " input contribution Wxh * x "
        1 to: inSize do: [:i |
            | wIndex |
            wIndex := (hIdx - 1) * inSize + i.
            sum := sum + (Wxh at: wIndex) * (inputVector at: i).
        ].
        " hidden contribution Whh * lastHidden "
        1 to: hSize do: [:hh |
            | wIndex |
            wIndex := (hIdx - 1) * hSize + hh.
            sum := sum + (Whh at: wIndex) * (lastHidden at: hh).
        ].

        " add bias "
        sum := sum + (bh at: hIdx).

        " apply activation "
        hiddenSum at: hIdx put: (activation value: sum).
    ].

    lastHidden := hiddenSum. "Update stored hidden"

    "Optionally compute output if outputSize > 0"
    outputSum := Array new: oSize withAll: 0.0.
    1 to: oSize do: [:outIdx |
        | sum |
        sum := 0.0.
        1 to: hSize do: [:hIdx |
            | wIndex |
            wIndex := (outIdx - 1) * hSize + hIdx.
            sum := sum + (Why at: wIndex) * (hiddenSum at: hIdx).
        ].
        sum := sum + (by at: outIdx).
        outputSum at: outIdx put: sum  "Typically you'd apply an output activation if needed" .
    ].

    lastOutput := outputSum.
    ^ outputSum  "Return the output of this time step"

]

{ #category : 'initialization' }
RNNLayer >> initialize [ 

    super initialize.
    Wxh := #() asFloatArray.
    Whh := #() asFloatArray.
    Why := #() asFloatArray.
    bh := #() asFloatArray.
    by := #() asFloatArray.

    lastInput := #() asFloatArray.
    lastHidden := #() asFloatArray.
    lastOutput := #() asFloatArray.
]

{ #category : 'initialization' }
RNNLayer >> inputSize: inSize hiddenSize: hidSize outputSize: outSize activation: anActivation [

    | randomScale totalWxh totalWhh totalWhy |
    inputSize := inSize.
    hiddenSize := hidSize.
    outputSize := outSize.
    activation := anActivation.
    randomScale := 0.01.

    "Allocate and initialize Wxh, Whh, Why, biases"

    totalWxh := inputSize * hiddenSize.
    totalWhh := hiddenSize * hiddenSize.
    totalWhy := hiddenSize * outputSize.

    Wxh := ((1 to: totalWxh)
        collect: [:i | (Random new next - 0.5) * randomScale ]) asArray asFloatArray.

    Whh := ((1 to: totalWhh)
        collect: [:i | (Random new next - 0.5) * randomScale ]) asArray asFloatArray.

    Why := ((1 to: totalWhy)
        collect: [:i | (Random new next - 0.5) * randomScale ]) asArray asFloatArray.

    bh := (1 to: hiddenSize) collect: [:i | (Random new next - 0.5) * randomScale ].
    by := (1 to: outputSize) collect: [:i | (Random new next - 0.5) * randomScale ].

    "Initialize hidden state to zero (or random if you prefer)."
    lastHidden := (1 to: hiddenSize) collect: [:i | 0.0 ].
    
    ^ self
]

{ #category : 'initialization' }
RNNLayer >> updateWeights: dLoss_dHidden learningRate: lr [ 

    "Use stored lastInput, lastHidden, dLoss_dHidden, etc. to do a gradient descent step."

    | inSize hSize oSize |

    inSize := inputSize.
    hSize := hiddenSize.
    oSize := outputSize.

    "1) Update Why, by using lastHidden and dLoss/dOut in lastOutput if you had it, or store that."
    "   Typically you'd get dLoss_dOut from the backward: method parameter.
     "

    "2) Update Wxh, Whh, bh using the chain from dLoss_dHidden."

    1 to: hSize do: [:hIdx |
        | gradHidden |
        gradHidden := (dLoss_dHidden at: hIdx).

        "Update bias bh"
        bh at: hIdx put: ((bh at: hIdx) - (lr * gradHidden)).

        "Update Wxh"
        1 to: inSize do: [:i |
            | wIndex dW |
            wIndex := (hIdx - 1) * inSize + i.
            dW := (lastInput at: i) * gradHidden.
            Wxh at: wIndex put: ((Wxh at: wIndex) - (lr * dW)).
        ].

        "Update Whh"
        1 to: hSize do: [:hh |
            | wIndex dW |
            wIndex := (hIdx - 1) * hSize + hh.
            dW := (lastHidden at: hh) * gradHidden.  "Note: lastHidden is from t-1 or t? depends on your design"
            Whh at: wIndex put: ((Whh at: wIndex) - (lr * dW)).
        ].
    ].

    "Similar approach for output weights Why, by. 
     Typically you need dLoss_dOut, so consider storing it or pass it in from the caller."

    ^ self

]
