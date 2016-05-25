package com.example

import Hello.{DMD, DVD}

trait CostFunc {
	def apply(a : Double, y : Double) : Double
	def delta(a : Double, y : Double, z : Double, actPrime : (Double => Double)) : Double

	def matrixDelta(aMatrix : DMD, yMatrix : DMD, zMatrix : DMD, actPrime : (Double => Double)) : DMD = {
		val result = yMatrix.mapPairs {
			case ((r, c), y) => delta(aMatrix(r, c), y, zMatrix(r, c), actPrime)
		}

		return result
	}
}

object CostFuncs {

	class CrossEntropyCostFunc extends CostFunc {
		def apply(a : Double, y : Double) = -(y * Math.log(a) + ((1 - y) * Math.log(1 - a)))

		def delta(a : Double, y : Double, z : Double, actPrime : (Double => Double)) =
			if (actPrime(z) == (a * (1 - a))) (a - y) else (a - y) / (a * (1 - a)) * actPrime(z)
	}

	class MeanSquareErrorCostFunc extends CostFunc {
		def apply(a : Double, y : Double) = Math.pow(y - a, 2) /2

		def delta(a : Double, y : Double, z : Double, actPrime : (Double => Double)) : Double = (a - y) * actPrime(z)
	}

	val crossEntropy = new CrossEntropyCostFunc()

	val meanSquareError = new MeanSquareErrorCostFunc()
}