package com.mocuml.net

import Hello.{ DMD, DVD }

trait Grad[A] {
    val weights : DMD

    val biases : DMD
}

case class Gradient[A](weights : DMD, biases : DMD) extends Grad[A]

trait GradBuilder[A <: Layer] {
    def buildGrad(delta : DMD, input : DMD) : Grad[A]

    def biasGrad(delta : DMD) : DMD

    def weightGrad(delta : DMD, input : DMD) : DMD
}

trait FullyConnectedGradBuilder extends GradBuilder[FullyConnectedLayer] {
    def biasGrad(delta : DMD) = sum(delta(*, ::))

    def weightGrad(delta : DMD, input : DMD) = delta * input.t

    def buildGrad(delta : DMD, input : DMD) = Gradient(weights = weightGrad(delta, input), biases = biasGrad(delta))

}

trait ConvolutionalGradBuilder extends GradBuilder[ConvolutionalLayer] { this : ConvolutionalLayer =>
    def biasGrad(delta : DMD) = {
        val numberOfInputs = delta.cols
        val reshaped = delta.reshape(hiddenArea, featureMaps * numberOfInputs)

        // Creates a matrix with bias gradients for each feature map along the columns.

        // If there are multiple inputs, the gradients for an input begins after
        // the gradients for all of the feature maps of the previous input have been
        // enumerated.
        val unshapedGrads1 = sum(delta(::, *))

        // Reshapes such that featureMaps increase along row axis, inputs increase
        // along col axis.
        val shapedGrads = unshapedGrads1.reshape(featureMaps, numberOfInputs)

        // Sums the gradients for all the inputs, returns row-vector storing gradient for all feature maps
        return sum(shapedGrads(*, ::))
    }

    def weightGrad(delta : DMD, input : DMD) = {
        val numberOfInputs = delta.cols
        val d = delta.reshape(featureMaps * numberOfInputs, hiddenArea)

        // Results in matrix with feature map changing along the row axis and gradients for each weight along the col axis

        // If there are multiple inputs, the gradients for an input begins after
        // the gradients for all of the feature maps of the previous input have been
        // enumerated.
        val grad = d * format(input)


        def recurse(value : DMD) : DMD = {
            if (value.rows <= featureMaps)
            {
                return d
            }
            else
            {
                val nD = value(0 until featureMaps, ::) + value(featureMaps until 2 * featureMaps, ::)

                return recurse(nD)
            }
        }

        return recurse(d)
    }

    def buildGrad(delta : DMD, input DMD) = Gradient(weights = weightGrad(delta, input), biases = biasGrad(delta))
}