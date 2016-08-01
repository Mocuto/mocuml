package com.mocuml.learning

import breeze.linalg._

import com.mocuml.Hello._
import com.mocuml.net._

object TrainLay {
	def apply[A <: Layer with UsesWeights[A]](layer : A, trainingParameters : TrainingParameters) : TrainLay[A] = TrainLay(layer, trainingParameters, Gradient.zero[A](layer))
}

case class TrainLay[A <: Layer with UsesWeights[A]](layer : A, trainingParameters : TrainingParameters, velocity : Grad[A]) {

	def update(grad : Grad[_], batchSize : Int, lambdaPerSize : Double) : TrainLay[A] = {
		val newVel = Gradient[A](
			weights = (velocity.weights * trainingParameters.momentum) - (grad.weights * (trainingParameters.learningRate / batchSize)),
			biases = (velocity.biases * trainingParameters.momentum) - (grad.biases * (trainingParameters.learningRate / batchSize))
		)

		val newLay = layer.gen(
			weights = (layer.weights * (1.0 - lambdaPerSize)) + newVel.weights,
			biases = layer.biases + newVel.biases
		)

		return TrainLay(newLay, trainingParameters, newVel)
	}
}