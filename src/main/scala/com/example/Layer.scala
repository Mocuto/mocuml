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
	def delta(l : Layer, d : DMD, z : DMD) : DMD

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

	def delta(l : Layer, d : DMD, z : DMD) : DMD = {
		return l.deltaByWeight(d) :* (z map (fPrime(_)))
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

	private def shapeFromDimensions(dim : Seq[Int]) : (Int, Int) = {
		return ((dim.dropRight(1)).fold(1)(_ * _), dim.last)
		/*if (dim.length == 1)
		{
			return (dim(0), 1)
		}
		else
		{
			return ((dim.dropRight(1)).fold(1)(_ * _), dim.last)
		}*/
	}

	lazy val inputShape = shapeFromDimensions(inputDimensions)

	lazy val lrfShape = shapeFromDimensions(lrfDimensions)

	lazy val hiddenShape = shapeFromDimensions(hiddenDimensions)


	/** 
		Formats the delta to be used during the backprop algorithm

		Expects input  to be in linear column format, ordered in ascending order by position within original data, with inner dimensions ascending before outer ones

		Formats it to be a matrix with dimensions: [localReceptiveFieldArea x number of input units]
	*/
	def deltaFormat(delta : DMD) : DMD = {
		val square  = delta.reshape(hiddenShape._1, hiddenShape._2)

		def recurse(r : Int, c : Int, accum : Option[DMD]) : Option[DMD] = {
			val (nextCol, lastCol) = (r > inputShape._1 - 1, c >= inputShape._2 - 1)
			if (lastCol && nextCol)
			{
				return accum
			}
			else if (nextCol)
			{
				return recurse(0, c + slide, accum)
			}

			val slice = ({
				val (overflowR, overflowC) = (r - lrfShape._1 + 1, c - lrfShape._2 + 1)

				val zeros = DenseMatrix.zeros[Double](lrfShape._1, lrfShape._2)

				val (r0, r1) = (lrfShape._1 - (r + 1), lrfShape._1 * 2 - (r + 2))
				val (c0, c1) = (lrfShape._2 - (c + 1), lrfShape._2 * 2 - (c + 2))

				/*if(overflowR < 0 || overflowC < 0)
				{
					
					zeros((lrfShape._1 - (r + 1)) until lrfShape._1, (lrfShape._2 - (c + 1)) until lrfShape._2) := square(0 to r, 0 to c)
					//zeros((lrfShape._1 - 1) to (lrfShape._1 - (r + 1)) by -1, (lrfShape._2 - 1) to (lrfShape._2 - (c + 1)) by -1) := square(0 to r, 0 to c)
				}
				else
				{
					zeros(0 to (lrfShape._1 - (r - hiddenShape._1)), 0 to (lrfShape._2 - (c - hiddenShape._2))) := square(
						(r - lrfShape._1) to math.max(r, hiddenShape._1),
						(c - lrfShape._2) to math.max(c, hiddenShape._2))
				}*/
				val dstRangeR = math.max(0, r0) to math.min(lrfShape._1 - 1, r1)
				val dstRangeC = math.max(0, c0) to math.min(lrfShape._2 -1, c1)
				val srcRangeR = math.max(0, r - (lrfShape._1 - 1)) to math.min(hiddenShape._1 - 1, r)
				val srcRangeC  = math.max(0, c - (lrfShape._2 - 1)) to math.min(hiddenShape._2 - 1, c)

				zeros(dstRangeR, dstRangeC) := square(
					srcRangeR,
					srcRangeC
				)
				zeros
			}).reshape(lrfShape._1 * lrfShape._2, 1)

			val next = accum match {
				case None => Some(slice)
				case Some(value : DMD) => Some(DenseMatrix.horzcat(value, slice))
			}

			return recurse(r + slide, c, next)
		}

		val unflipped = recurse(0, 0, None).getOrElse(DenseMatrix.zeros[Double](lrfShape._1 * lrfShape._2, inputShape._1 * inputShape._2))
		
		return flipud(unflipped)
	}

	def format(i : DMD) : DMD = {

		val square = i.reshape(inputShape._1, inputShape._2)

		def recurse(r : Int, c : Int, accum : DMD) : DMD = {
			//TODO: Implement for batch inputs
			val (lastCol, nextCol) = (c + lrfShape._2 >= inputShape._2, r + lrfShape._1 > inputShape._1)
			if(lastCol && nextCol) {
				return accum
			}
			if(nextCol) {
				return recurse(0, c + 1, accum)
			}
			val next = DenseMatrix.horzcat(accum, square(r until (r + lrfShape._1), c until (c + lrfShape._2)).reshape(lrfShape._1 * lrfShape._2, 1))

			return recurse(r + slide, c, next)
		}
		val start = square(0 until lrfShape._1, 0 until lrfShape._2).reshape(lrfShape._1 * lrfShape._2, 1)
		return recurse(1, 0, start)
	}

	/**
		Assumes that each row corresponds to a different featuremap

		Returns the data in a linear format, where outputs from the same feature map are next to each other.

		(e.g., matrix with feature maps a, b, and c),

		i,
		a1  a2  a3
		b1  b2  b3
		c1  c2  c3

		returns,
		a1
		a2
		a3
		b1
		b2
		b3
		c1
		c2
		c3
	*/
	def unformat(i : DMD) : DMD = {
		return i.t.reshape(i.rows * i.cols, 1, false)
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

	val hiddenArea = hiddenShape._1 * hiddenShape._2

	def copy(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : ConvolutionalLayer = this.copy(weights, biases, f, fPrime)

	def feedforward(input : DMD) : (DMD, DMD) = {
		//val wA = weights * format(input)
		//val z = wA(::, *) + biases

		//return (z, z.map(f(_)))

		val wA = format(input).t * weights.t
		val z = wA(*, ::) + biases

		return (z.reshape(hiddenArea, 1), z.map(f(_)).reshape(hiddenArea, 1))
	}

	/**
		Expects delta in a linear format
	*/
	def deltaByWeight(delta : DMD) : DMD = {
		val hu = hiddenShape._1 * hiddenShape._2 // Number of hidden units per feature map
		return (0 until featureMaps *  hu by hu).foldLeft(new DenseMatrix[Double](0,1)) { (accum, i) => //This map will be performed for each feature map
			val d = delta(i until (i + hu), 0 to 0) // Slice the hidden units from one feature map

			val dbd = (weights(i / hu to i / hu, ::) * deltaFormat(d))
			DenseMatrix.vertcat(accum, dbd.reshape(dbd.rows * dbd.cols, 1))
		}
	}

	def delta(l : Layer, d : DMD, z : DMD) : DMD = {
		return l.deltaByWeight(d) :* (z map (fPrime(_)))
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