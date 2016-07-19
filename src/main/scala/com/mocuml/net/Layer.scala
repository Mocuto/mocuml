package com.example

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

import scala.annotation.tailrec

//import scala.concurrent.{ Future, Promise }
//import scala.concurrent.ExecutionContext.Implicits.global
//import scala.util.Success

//import CostFuncs._

import Hello.{ DMD, DVD, timeThis }

object Layer {

	def unapply(l : Layer) = Option((l.weights, l.biases, l.f, l.fPrime))
}

trait Layer {

	val weights : DMD

	val biases : DVD

	val f : (Double => Double)

	val fPrime : (Double => Double)

	def deltaByWeight(delta : DMD) : DMD

	def feedforward(input : DMD) : (DMD, DMD) // First DMD is the output, Second DMD is the activation, a = sigmoid(z)

	def gradientBiases(delta : DMD) : DMD

	def gradientWeights(delta : DMD, previousActivation : DMD) : DMD

	def update(delta : DMD, lastActivation : DMD) : Layer

	/**
		Calculates the delta for this layer,

		output - An assumed "z" that this layer produced
		lastDelta - The delta for the layer after this one (which precedes this layer in the backprop algo)
	*/
	def delta(l : Layer, d : DMD, z : DMD) : DMD

	def add(that : Layer) = gen(weights + that.weights, biases + that.biases, f, fPrime)

	def +(that : Layer) = add(that)

	def minus(that : Layer) = gen(weights - that.weights, biases - that.biases, f, fPrime)

	def -(that : Layer) = minus(that)

	def mult(const : Double) = gen(weights * const, biases * const, f, fPrime)

	def **(const : Double) = mult(const)

	def gen(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : Layer

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

	def gradiantBias(delta : DMD) : Layer = return delta

	def gen(weights : DMD = weights, biases : DVD = biases, f : (Double => Double) = f, fPrime : (Double => Double) = fPrime) = copy(weights, biases, f, fPrime)
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
	val stride : Int

	lazy val hiddenDimensions : Seq[Int] = ((inputDimensions zip(lrfDimensions)) map { case (iD, lrfD) => 1 + ((iD - lrfD) / stride) })

	private def shapeFromDimensions(dim : Seq[Int]) : (Int, Int) = {
		//return ((dim.dropRight(1)).fold(1)(_ * _), dim.last)
		return (dim.head, dim.drop(1).fold(1)(_ * _))
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
		val noOfInputs = delta.cols
		val square  = delta.reshape(hiddenShape._1, hiddenShape._2 * noOfInputs)

		def recurse(r : Int, c : Int, accum : Option[DMD], inputNum : Int) : DMD = {
			val (nextCol, lastCol) = (r > inputShape._1 - 1, c >= inputShape._2 - 1)
			if (inputNum >= noOfInputs)
			{
				return accum getOrElse new DenseMatrix[Double](0, 0)
			}
			else if (lastCol && nextCol)
			{
				return recurse(0, 0, accum, inputNum + 1)
			}
			else if (nextCol)
			{
				return recurse(0, c + stride, accum, inputNum)
			}
			else
			{
				val slice = ({
					val (overflowR, overflowC) = (r - lrfShape._1 + 1, c - lrfShape._2 + 1)

					val zeros = DenseMatrix.zeros[Double](lrfShape._1, lrfShape._2)

					val (r0, r1) = (lrfShape._1 - (r + 1), lrfShape._1 * 2 - (r + 2))
					val (c0, c1) = (lrfShape._2 - (c + 1), lrfShape._2 * 2 - (c + 2))

					val cOffset = inputNum * inputShape._2

					val dstRangeR = math.max(0, r0) to math.min(lrfShape._1 - 1, r1)
					val dstRangeC = math.max(0, c0) to math.min(lrfShape._2 -1, c1)
					val srcRangeR = math.max(0, r - (lrfShape._1 - 1)) to math.min(hiddenShape._1 - 1, r)
					val srcRangeC  = (cOffset + math.max(0, c - (lrfShape._2 - 1))) to (cOffset + math.min(hiddenShape._2 - 1, c))

					zeros(dstRangeR, dstRangeC) := square(
						srcRangeR,
						srcRangeC
					)
					zeros
				}).reshape(lrfShape._1 * lrfShape._2, 1)

				val next = accum.map(s => DenseMatrix.horzcat(s, slice))

				return recurse(r + stride, c, Some(next getOrElse (slice)), inputNum)
			}

		}

		val unflipped = recurse(0, 0, None, 0)

		return flipud(unflipped)
	}

	final def format(i : DMD) : DMD = {

		val noOfInputs = (i.rows * i.cols) / (inputShape._1 * inputShape._2)
		val square = i.reshape(inputShape._1, inputShape._2 * noOfInputs)

		println("format")

		//@tailrec
		def recurse(r : Int, c : Int, accum : Option[DMD], inputNum : Int) : DMD = {
			//TODO: Implement for batch inputs
			val (lastCol, nextCol) = (c + lrfShape._2 >= inputShape._2, r + lrfShape._1 > inputShape._1)

			if (inputNum >= noOfInputs)
			{
				return timeThis("recurse base case") { accum getOrElse new DenseMatrix[Double](0,0) }
			}
			else if (lastCol && nextCol)
			{
				return timeThis(s"recurse input $inputNum") { recurse(0, 0, accum, inputNum + 1) }
			}
			else if (nextCol)
			{
				return recurse(0, c + stride, accum, inputNum)
			}
			else
			{
				val iC = c + (inputNum * inputShape._2)
				val slice = square(r until (r + lrfShape._1), iC until (iC + lrfShape._2)).reshape(lrfShape._1 * lrfShape._2, 1, View.Copy)
				val next = accum map (s => DenseMatrix.horzcat(s, slice))

				return recurse(r + stride, c, Some(next getOrElse (slice)), inputNum)
			}
		}
		return timeThis("formatRecurse"){recurse(0, 0, None, 0)}
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
		val noOfInputs = i.cols / (hiddenShape._1 * hiddenShape._2)
		return i.t.reshape(hiddenShape._1 * hiddenShape._2 * lrfShape._1 * lrfShape._2, noOfInputs, false)
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

	val stride = 1

	val hiddenArea = hiddenShape._1 * hiddenShape._2

	def gen(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : ConvolutionalLayer = this.copy(weights, biases, f, fPrime)

	/*
		Expects input in linear format, where each column is a separate input
	*/
	def feedforward(input : DMD) : (DMD, DMD) = {

		val wA = format(input).t * weights.t
		val z = wA(*, ::) + biases

		val noOfInputs = input.cols

		println(s"feedforward conv $noOfInputs $hiddenArea ${z.rows} ${z.cols}")

		val zShaped = z.reshape(hiddenArea * featureMaps, noOfInputs)

		println("feedforward conv")

		return (zShaped, zShaped.map(f(_)))
	}

	/**
		Expects delta in a linear format
	*/
	def deltaByWeight(delta : DMD) : DMD = {
		val noOfInputs = delta.cols
		val hu = hiddenShape._1 * hiddenShape._2 // Number of hidden units per feature map
		val iu = inputShape._1 * inputShape._2;
		return (0 until featureMaps *  hu by hu).foldLeft(DenseMatrix.zeros[Double](iu, noOfInputs)/*new DenseMatrix[Double](0,1)*/) { (accum, i) => //This map will be performed for each feature map
			val d = delta(i until (i + hu), 0 until noOfInputs) // Slice the hidden units from one feature map

			val dbd = (weights(i / hu to i / hu, ::) * deltaFormat(d))
			//DenseMatrix.vertcat(accum, dbd.reshape(dbd.rows * dbd.col, noOfInputs))
			accum + dbd.reshape(iu, noOfInputs)
		}
	}

	def delta(l : Layer, d : DMD, z : DMD) : DMD = {
		return l.deltaByWeight(d) :* (z map (fPrime(_)))
	}
}

object MaxPoolLayer {
	def gen(lrfDimensions : Seq[Int], inputDimensions : Seq[Int]) = 
		MaxPoolLayer(new DenseMatrix[Double](0,0), DenseVector[Double](), identity, identity, lrfDimensions, inputDimensions)
}

case class MaxPoolLayer(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double), lrfDimensions : Seq[Int], inputDimensions : Seq[Int]) extends Layer with LocalReceptiveFieldFormatter
{
	private var argmaxes = Array.empty[Int]

	val stride = 2
	override lazy val hiddenDimensions : Seq[Int] = 
		((inputDimensions.take(2) zip(lrfDimensions.take(2))) map { case (iD, lrfD) => 1 + ((iD - lrfD) / stride) }) ++ 
			List.tabulate(inputDimensions.size - 2) { i => inputDimensions(i + 2)}

	val hiddenArea = hiddenShape._1 * hiddenShape._2

	println(s"$hiddenDimensions")

	def gen(weights : DMD, biases : DVD, f : (Double => Double), fPrime : (Double => Double)) : MaxPoolLayer = this.copy(weights, biases, f, fPrime)

	def feedforward(input : DMD) : (DMD, DMD) = {

		val noOfInputs = input.cols
		val x = format(input)
		
		argmaxes = argmax(x(::, *)).t.data
		println(s"maxpool feedforward $hiddenArea $noOfInputs ${x.rows} ${x.cols}")
		println(max(x(::, *)).t.asDenseMatrix)
		val maxes = max(x(::, *)).t.asDenseMatrix.reshape(hiddenArea, noOfInputs, View.Copy)
		println("maxes")
		return (maxes, maxes)
	}

	def delta(l : Layer, d : DMD, z : DMD) : DMD = return l.deltaByWeight(d)

	def deltaByWeight(delta : DMD) : DMD = {
		val d = DenseMatrix.zeros[Double](inputShape._1, inputShape._2)
		for((maxIndex, fieldIndex) <- argmaxes.zipWithIndex) {
			val r = (fieldIndex % hiddenShape._1) + (maxIndex % lrfShape._1)
			val c = (fieldIndex / hiddenShape._1) + (maxIndex / lrfShape._1)
			d(r to r, c to c) := delta(fieldIndex, 0)
		}
		return d.reshape(d.rows * d.cols, 1)
	}
}

case class SoftMaxLayer(weights : DMD, biases : DVD, f : (Double => Double) = identity, fPrime : (Double => Double) = identity) extends Layer {
	def deltaByWeight(delta : DMD) : DMD = return weights.t * delta

	def feedforward(input : DMD) : (DMD, DMD) = {
		val wA = weights * input
	  val z = wA(::, *) + biases;

	  return (z, activate(z))
	}

	def activate(z : DMD) : DMD = {
		val expZ = exp(z)
		val summed = sum(expZ(::, *))
		val divided = expZ.mapPairs { case ((_, c), x) => x / summed(c) }

		return divided
	}

	def actPrime(z : DMD) : DMD = {
		val expZ = exp(z)
		val summed = sum(expZ(::, *))
		val summedSquared = pow(summed, 2.0)

		return expZ.mapPairs { case ((_,c), x) => x * (summed(c) - x) / summedSquared(c) }
	}

	def delta(l : Layer, d : DMD, z : DMD) : DMD = {
		return l.deltaByWeight(d) :* actPrime(z)
	}

	def gen(weights : DMD = weights, biases : DVD = biases, f : (Double => Double) = f, fPrime : (Double => Double) = fPrime) = copy(weights, biases, f, fPrime)

}