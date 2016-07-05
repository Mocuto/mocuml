package com.example

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

//import scala.annotation.tailrec

//import scala.concurrent.{ Future, Promise }
//import scala.concurrent.ExecutionContext.Implicits.global
//import scala.util.Success

//import CostFuncs._

import Hello.{ DMD, DVD }

object Layer {

	def unapply(l : Layer) = Option((l.weights, l.biases, l.f, l.fPrime))
}

trait Layer { this: { def copy(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : Layer} =>

	val weights : DMD

	val biases : DVD

	val f : (Double => Double)

	val fPrime : (Double => Double)

	def deltaByWeight(delta : DMD) : DMD

	def feedforward(input : DMD) : (DMD, DMD) // First DMD is the output, Second DMD is the activation, a = sigmoid(z)

	/**
		Calculates the delta for this layer,

		output - An assumed "z" that this layer produced
		lastDelta - The delta for the layer after this one (which precedes this layer in the backprop algo)
	*/
	def delta(output : DMD, lastDelta : DMD, lastWeights : DMD) : DMD

	def add(that : Layer) = gen(weights + that.weights, biases + that.biases, f, fPrime)

	def +(that : Layer) = add(that)

	def minus(that : Layer) = gen(weights - that.weights, biases - that.biases)

	def -(that : Layer) = minus(that)

	def mult(const : Double) = gen(weights * const, biases * const)

	def **(const : Double) = mult(const)

	def gen(weights : DMD = weights, biases : DVD = biases, f : (Double => Double) = f, fPrime : (Double => Double) = fPrime) : Layer = copy(weights, biases, f, fPrime)


}

object FullyConnectedLayer {
	type WeightInit = (Int, Int) => DMD
	type BiasInit = Int => DVD

	private def defaultWeightInit(size : Int, numOfInputs : Int) = DenseMatrix.rand(size, numOfInputs, Rand.gaussian)
	private def defaultBiasInit(size : Int) = DenseVector.rand(size, Rand.gaussian)

	def gen(size : Int, numOfInputs : Int, f : (Double => Double), fPrime : (Double => Double),
			weightInit : WeightInit = defaultWeightInit,
			biasInit : BiasInit = defaultBiasInit) : FullyConnectedLayer = FullyConnectedLayer(
		weights = weightInit(size, numOfInputs),
		biases = biasInit(size),
		f = f,
		fPrime = fPrime
	)
}

case class FullyConnectedLayer(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) extends Layer {

	def deltaByWeight(delta : DMD) : DMD = return weights.t * delta

	def feedforward(input : DMD) : (DMD, DMD) = {
		val wA = weights * input
	  val outputs = wA(::, *) + biases;
	  return (outputs, outputs.map(x => f(x)))
	}

	def delta(output : DMD, lastDelta : DMD, lastWeights : DMD) : DMD = {
		return (lastWeights.t * lastDelta) :* (output map (z => fPrime(z)))
	}
}

trait InputFormatter {
	def format(i : DMD) : DMD
	def unformat(i : DMD) : DMD
}

object LocalReceptiveFieldFormatter {
	def getInputMatrixDimensions(lrfDimensions : Seq[Int], inputDimensions : Seq[Int], slide : Int) = ((1, 1) /: lrfDimensions.zip(inputDimensions)) {
		case ((lrfVol, slideVol), (lrf, input)) => (lrfVol * lrf, slideVol * ((input - lrf) / slide + 1))
	}
}

trait LocalReceptiveFieldFormatter extends InputFormatter {

	val lrfDimensions : Seq[Int]
	val inputDimensions : Seq[Int]
	val slide = 1

	val hiddenDimensions : Seq[Int] = ((inputDimensions zip(lrfDimensions)) map { case (iD, lrfD) => 1 + (iD - lrfD) })

	lazy val inputShape = if (inputDimensions.length == 1) (inputDimensions(0), 1) else {
		(inputDimensions(1), inputDimensions(0) * ((1 /: inputDimensions.slice(2, inputDimensions.length))(_ * _)))
	}

	lazy val lrfShape = if (lrfDimensions.length == 1) (lrfDimensions(0), 1) else {
		(lrfDimensions(1), lrfDimensions(0) * ((1 /: lrfDimensions.slice(2, lrfDimensions.length))(_ * _)))
	}

	lazy val hiddenShape = if (hiddenDimensions.length == 1) (hiddenDimensions(0), 1) else {
		(hiddenDimensions(1), hiddenDimensions(0) * ((1 /: hiddenDimensions.slice(2, hiddenDimensions.length))(_ * _)))
	}

	def deltaFormat(delta : DMD) : DMD = {
		val square  = delta.reshape(hiddenShape._1, hiddenShape._2)

		def recurse(r : Int, c : Int, accum : Option[DMD]) : Option[DMD] = {
			val (lastRow, nextRow) = (r >= hiddenShape._1 - 1, c > hiddenShape._2)
			if (lastRow && nextRow)
			{
				return accum
			}
			else if (nextRow)
			{
				return recurse(r + slide, 0, accum)
			}

			val slice = {
				val (overflowR, overflowC) = (r - lrfShape._1 + 1, c - lrfShape._2 + 1)

				if(overflowR < 0 || overflowC < 0)
				{
					val zeros = DenseMatrix.zeros[Double](lrfShape._1, lrfShape._2)
					
					zeros((lrfShape._1 - (r + 1)) until lrfShape._1, (lrfShape._2 - (c + 1)) until lrfShape._2) := square(0 to r, 0 to c)
				}
				else
				{
					square((r - lrfShape._1) to r, (c - lrfShape._2) to c)
				}
			}

			val next = accum match {
				case None => Some(slice)
				case Some(value : DMD) => Some(DenseMatrix.horzcat(value, slice))
			}

			return recurse(r, c + slide, next)
		}
		val numOfSlides = (hiddenShape._1 + lrfShape._1 - 1) * (hiddenShape._2 + lrfShape._2 - 1)
		return (recurse(0, 0, None)).getOrElse(DenseMatrix.zeros[Double](lrfShape._1 * lrfShape._2, numOfSlides))
	}

	def format(i : DMD) : DMD = {

		val square = i.reshape(inputShape._1, inputShape._2)

		def recurse(r : Int, c : Int, accum : DMD) : DMD = {
			//TODO: Implement for batch inputs
			val (lastRow, nextRow) = (r + lrfShape._1 >= inputShape._1, c + lrfShape._2 > inputShape._2)
			if(lastRow && nextRow) {
				return accum
			}
			if(nextRow) {
				return recurse(r + slide, 0, accum)
			}
			val next = DenseMatrix.horzcat(accum, square(r until (r + lrfShape._1), c until (c + lrfShape._2)).reshape(lrfShape._1 * lrfShape._2, 1))

			return recurse(r, c + slide, next)
		}
		val start = square(0 until lrfShape._1, 0 until lrfShape._2).reshape(lrfShape._1 * lrfShape._2, 1)
		return recurse(0, 1, start)
	}

	def unformat(i : DMD) : DMD = {
		return i.reshape(i.rows * i.cols, 1)
	}
}

object ConvolutionalLayer {

	type ConvolutionalWeightInit = (Seq[Int], Int) => DMD
	type ConvolutionalBiasInit = Int => DVD

	def defaultWeightInit(lrfDimensions : Seq[Int], numOfFeatureMaps : Int) : DMD = DenseMatrix.rand[Double](numOfFeatureMaps, lrfDimensions.reduce(_ * _))
	def defaultBiasInit(numberOfFeatureMaps : Int) : DVD = DenseVector.rand[Double](numberOfFeatureMaps)

	def gen(
		lrfDimensions : Seq[Int],
		inputDimensions : Seq[Int],
		f : (Double => Double),
		fPrime : (Double => Double),
		numOfFeatureMaps : Int,
		weightInit : ConvolutionalWeightInit = defaultWeightInit,
		biasInit : ConvolutionalBiasInit = defaultBiasInit
	) : ConvolutionalLayer = ConvolutionalLayer(
		weightInit(lrfDimensions, numOfFeatureMaps),
		biasInit(numOfFeatureMaps),
		f,
		fPrime,
		lrfDimensions,
		inputDimensions,
		numOfFeatureMaps
	)
}

case class ConvolutionalLayer(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double),
		lrfDimensions : Seq[Int], //Dimensions for the local receptive field
		inputDimensions : Seq[Int],
		featureMaps : Int) extends Layer with LocalReceptiveFieldFormatter {
	def copy(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : ConvolutionalLayer = this.copy(weights, biases, f, fPrime)

	def feedforward(input : DMD) : (DMD, DMD) = {
		val wA = timesWeight(input)
		val z = wA(::, *) + biases

		return (z, z.map(f(_)))
	}

	def deltaByWeight(delta : DMD) : DMD = {
		val hu = hiddenDimensions._1 * hiddenDimensions._2 // Number of hidden units per feature map
		(0 until featureMaps *  hu by hu).map { i => //This map will be performed for each feature map
			val d = delta(i until i + hu) // Slice the hidden units from one feature map

			val dbd = weights(i / hu, ::) * deltaFormat(d)
			//TODO: Take the result of this map and concatonate into one long column
		}
	}

	def delta(output : DMD, lastDelta : DMD, lastWeights : DMD) : DMD = {
		println("Implment delta in ConvolutionalLayer")
		return output
	}
}

/*object PoolLayer {
	def gen(
		layerDimensions : Seq[Int],
		inputDimensions : Seq[Int],
		f : (Double => Double),
		fPrime : (Double => Double),
		numOfFeatureMaps : Int,
	)
}*/