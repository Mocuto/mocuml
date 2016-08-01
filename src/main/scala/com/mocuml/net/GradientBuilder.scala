package com.mocuml.net

import com.mocuml.Hello.{ DMD, DVD }
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

trait Grad[A <: Layer] {
  val weights : DMD

  val biases : DVD
}

object Gradient {
  def zero[A <: Layer with UsesWeights[A]](l : A) : Grad[A] = 
    Gradient[A](
      weights = DenseMatrix.zeros[Double](l.weights.rows, l.weights.cols),
      biases = DenseVector.zeros[Double](l.biases.length)
    )
}

case class Gradient[A <: Layer](weights : DMD, biases : DVD) extends Grad[A]

trait GradBuilder[A <: Layer] {
  def buildGrad(delta : DMD, input : Activation) : Grad[A]

  def biasGrad(delta : DMD) : DVD

  def weightGrad(delta : DMD, input : Activation) : DMD
}

trait FullyConnectedGradBuilder extends GradBuilder[FullyConnectedLayer] {
  def biasGrad(delta : DMD) = sum(delta(*, ::))

  def weightGrad(delta : DMD, input : Activation) = delta * input.a.t

  def buildGrad(delta : DMD, input : Activation) =
    Gradient[FullyConnectedLayer](weights = weightGrad(delta, input), biases = biasGrad(delta))

}

trait ConvolutionalGradBuilder extends GradBuilder[ConvolutionalLayer] { this : ConvolutionalLayer =>
  def biasGrad(delta : DMD) : DVD = {
    val numberOfInputs = delta.cols
    val reshaped = delta.reshape(hiddenArea, featureMaps * numberOfInputs)

    // Creates a matrix with bias gradients for each feature map along the columns.

    // If there are multiple inputs, the gradients for an input begins after
    // the gradients for all of the feature maps of the previous input have been
    // enumerated.
    val unshapedGrads1 = sum(delta(::, *))

    // Reshapes such that featureMaps increase along row axis, inputs increase
    // along col axis.
    val shapedGrads = unshapedGrads1.t.asDenseMatrix.reshape(featureMaps, numberOfInputs)

    // Sums the gradients for all the inputs, returns row-vector storing gradient for all feature maps
    return sum(shapedGrads(*, ::))
}

  def weightGrad(delta : DMD, input : Activation) : DMD = {
    val numberOfInputs = delta.cols
    val d = delta.reshape(featureMaps * numberOfInputs, hiddenArea)

    // Results in matrix with feature map changing along the row axis and gradients for each weight along the col axis

    // If there are multiple inputs, the gradients for an input begins after
    // the gradients for all of the feature maps of the previous input have been
    // enumerated.
    val grad = d * format(input.a)


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

  def buildGrad(delta : DMD, input : Activation) = Gradient[ConvolutionalLayer](weights = weightGrad(delta, input), biases = biasGrad(delta))

}