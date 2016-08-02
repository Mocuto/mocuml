package com.mocuml

import com.mocuml.net._
import com.mocuml.reporting._
import com.mocuml.learning._

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._

import scala.annotation.tailrec

import scala.concurrent.{ Future, Promise }
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Success

import scala.util.Random

import CostFuncs._
import net._

import com.mocuml.util._

object Hello {

	case class GradLayPair[A <: Layer](grad : Grad[A], lay : Layer)

  type DMD = DenseMatrix[Double]
  type DVD = DenseVector[Double]

  def timeThis[A](name : String)(func : => A) : A = {
  	val startTime = System.currentTimeMillis
  	val r = func
  	val endTime = System.currentTimeMillis
  	println(s"$name took ${endTime - startTime} millis")
  	return r
  }

  case class TrainingParameters(
  		batchSize : Int,
  		epochs : Int,
  		learningRate : Double,
  		momentum : Double = 0.0,
  		lmbda : Double = 0.0,
  		earlyStopCount : Double = Double.PositiveInfinity)

  case class TrainingData(exampleSet : TrainingBatch, testSet : TrainingBatch, validationSet : TrainingBatch)

  case class TrainingResult(
  		network : Network,
  		trainingData : TrainingData,
  		hyperparameters : TrainingParameters,
  		exampleAssessment : NetworkAssessment,
  		testAssessment : NetworkAssessment,
  		validationAssessment : NetworkAssessment)

  class Trainer[A <: CostFunc](val costFunc : A) {
  	self : Reporter with Assessor[A] =>
		final def train(n : Network, td : TrainingData, hp : TrainingParameters) : Future[TrainingResult] = {

			val p = Promise[TrainingResult]

			val tls = n.layers.map(l => l match {
				case uwl : UsesWeights[_]  => Some(uwl.genTrainLay(hp))
				case _ => None
			})

			@tailrec
			def inner(network : Network, trainlays : List[Option[TrainLay[_]]], epochsInner : Int, sinceLastLoss : Int, bestCost : Double) : Promise[TrainingResult] = {

					if (epochsInner <= 0 || sinceLastLoss >= hp.earlyStopCount)
					{
						val r = TrainingResult(
							network = network,
							trainingData = td,
							hyperparameters = hp,
							exampleAssessment = assess(network, td.exampleSet),
							testAssessment = assess(network, td.testSet),
							validationAssessment = assess(network, td.validationSet)
						)
						return p.success(r)
					}
					else
					{
						//TODO: Find a way to optimize shuffle
						val batches = timeThis("Shuffle"){(Random.shuffle(td.exampleSet).toVector).grouped(hp.batchSize).toList}

						val (newNet, newTrainLays) = whenEpochStarts(network, td, hp, epochsInner) {
							applyBatches(network, trainlays, batches, hp.lmbda / td.exampleSet.size)
						}

						//val netCost = totalCost(newNet, td.validationSet)
						println("Assessing validation")
						val NetworkAssessment(_, _, validationCost) = assess(newNet, td.validationSet)
						println("Finished assessing validation")
						val (newBestCost, newLastLoss) = if(validationCost < bestCost) (validationCost, 0) else (bestCost, sinceLastLoss + 1)


						//TODO: Add diagnostics call here
						return inner(newNet, newTrainLays, epochsInner - 1, newLastLoss, newBestCost)
					}
			}

			whenTrainingStarts(n, td, hp)(p.future)

			Future {
				inner(n, tls, hp.epochs, 0, Double.PositiveInfinity)
			}

			return p.future

		}

		@tailrec
		final def applyBatches(n : Network, traininglays : List[Option[TrainLay[_]]], batches : List[TrainingBatch], lambdaPerSize : Double)
			 	: (Network, List[Option[TrainLay[_]]]) = batches match {

			case List() => return (n, traininglays)
			case batch :: rest => {

				println("before backprop")
				val hasDeltas = n.backprop(batch, costFunc)
				println("after backprop")

				val (newTrainLays, newLayers) = ((List.empty[Option[TrainLay[_]]], List.empty[Layer]) /: ((traininglays.size - 1) to 0 by -1)) { case ((aTL, aL), i) =>
					val l = n.layers(i);

					(traininglays(i), hasDeltas(i)) match {
						case (Some(tl : TrainLay[_]), Some(g : Grad[_])) => {
							val newTl = tl.update(g, batch.size, lambdaPerSize)
						 (Some(newTl) :: aTL, newTl.layer :: aL)
						}
						case _ => (None :: aTL, l :: aL)
					}
				}

				return applyBatches(Network(newLayers.toList), newTrainLays.toList, rest, lambdaPerSize)

				/*
				//val velW = gradWs.zip(velocity.weights).map { case (gradW, vW) => (vW * momentum) - (gradW * (learningRate / batch.length)) }
				//val velB = gradBs.zip(velocity.biases).map { case (gradB, vB) => (vB * momentum) - (gradB * (learningRate / batch.length)) }

				val (newLayers, velLayers) = (for {
				 	((l, vl), (gradW, gradB)) <- (n.layers.zip(velocity.layers)).zip(gradWs.zip(gradBs));
				 	(w, b, _, _) = Layer.unapply(l).get;
				 	(vW, vB, _, _) = Layer.unapply(vl).get
				} yield {

					println(s"yield ${gradB.size}")
					val velLayer = vl.gen(
						weights = (vW * momentum) - (gradW * (learningRate / batch.length)),
						biases = (vB * momentum) - (gradB * (learningRate / batch.length)),
						f = vl.f,
						fPrime = vl.fPrime
					)
					println("velLayer")
					val newL = l.gen(
						weights = (w * (1.0 - lambdaPerSize)) + velLayer.weights,
						biases = b + velLayer.biases,
						f = l.f,
						fPrime = l.fPrime
					)
					println("newL")
					(newL, velLayer)
				}).unzip
				println("new layers, vellayers")

				val newVel = Network(velLayers)//Network(velW, velB, n.f, n.fPrime)
				val newN = Network(newLayers)
				/*val newN = Network(
					(n.weights.map(_ * (1.0 - lambdaPerSize)) ).zip(velW).map(w => w._1 + w._2),
					(n.biases.zip(velB)).map(b => b._1 + b._2),
					n.f,
					n.fPrime)*/

				return applyBatches(newN, rest, learningRate, momentum, lambdaPerSize, newVel)*/
			}
		}
  }



  object Network {

    def empty = Network(List())

    //def withIdentityActivation(layers : List[Layer]) : Network = Network(layers.map(l => l.gen(l.weights, l.biases, f = identity, fPrime = identity)))

    def withLayerSizes(sizes : List[Int], f : (Double => Double), fPrime : (Double => Double)) : Network =
    	Network((sizes.sliding(2) map { case List(a, b) => FullyConnectedLayer(
    		weights = DenseMatrix.rand(b, a, Rand.gaussian),
    		biases = DenseVector.rand(b, Rand.gaussian).map(_ - 0.5),
				f = f,
				fPrime = fPrime)
    	}).toList)

    	/*Network(
    		(sizes.sliding(2) map { case List(a, b) => DenseMatrix.rand(b, a, Rand.gaussian) } ).toList,
    		(sizes.tail map (size => DenseVector.rand(size, Rand.gaussian).map(_ - 0.5))).toList,
    		f,
    		fPrime)*/
  }

  //case class Network(weights : List[DMD], biases : List[DVD], f : (Double => Double), fPrime : (Double => Double)) {
	case class Network(layers : List[A forSome { type A <: Layer }]) {

	  def feedforward(input : List[Double]) : DMD = return feedforward(listToDVD(input))

	  def feedforward(input : DVD) : DMD = return (feedforwardAccum()(List((LayAct(input.asDenseMatrix.t))))).head.a

	  def feedforwardBatch(batch : TrainingBatch) : DMD = {
	  	val (inputMatrix, _) = batchToIOMatrices(batch)
	  	return (feedforwardAccum()(List((LayAct(inputMatrix)) ))).head.a
	  }

	  //previous activations is a list of tuple, the tuple being (output, activation)

		//final def feedforwardAccum(fromLayerIndex : Int = 0)(previousActivations : List[(DMD, DMD)]) : List[(DMD, DMD)] = {
		final def feedforwardAccum(fromLayerIndex : Int = 0)(previousActivations : List[Activation]) : List[Activation] = {
	    if(layers.isEmpty)
	    {
		  	return previousActivations
	    }
	    else
	    {
	    	val accumulated = (previousActivations /: layers.drop(fromLayerIndex)) ((accum, l) => {
	    		println("feedforwardAccum")
	    		l.feedforward(accum.head) :: accum
	    	})

	    	return accumulated
	    }
	  }

	  //def backprop[A <: CostFunc](batch : TrainingBatch, costFunc : A) : Network = {
	  def backprop[A <: CostFunc](batch : TrainingBatch, costFunc : A) : List[Option[Grad[_]]] = {
	  	//TODO: Throw exception on empty batch
	  	//TODO: Rewrite to use prepend rather than append

	    @tailrec
	    def calcGrads(
	    		layers : List[Layer],
	    		deltas : List[DMD],
	    		acts : List[Activation],
	    		grads : List[Option[Grad[_]]]) : List[Option[Grad[_]]] = (layers, deltas, acts) match {

			  case (_, _, _ :: List()) => return grads

			  //case ((restWeights :+ weight, headDelta :: _, restZ :+ z)) => {
			  case(((restLayers :+ upperLayer) :+ lowerLayer, lowerDelta :: _, act :: input :: restActs)) => {
			  	//val nextDelta = (w.t * headDelta) :*  (z map(x => fPrime(x)))
			  	val nextDelta = upperLayer.delta(lowerLayer, lowerDelta, act)

					val upperGrad = upperLayer match {
						case cbg : GradBuilder[_] => Some(cbg.buildGrad(nextDelta, input))
						case _ => None
					}
			  	println("upperGrad")
			  	println(upperGrad)
			  	return calcGrads(restLayers :+ upperLayer, nextDelta :: deltas, input :: restActs, upperGrad :: grads)
				}
		  }

		  val (iMatrix, oMatrix) = batchToIOMatrices(batch)

			val last :: secondToLast :: rest = feedforwardAccum(0)(List( LayAct(iMatrix)))
			println("backprop feedforward")
			//val (z, a) = (last.z, last.a)

			//val (restZ, restA) = ((List.empty[DMD], List.empty[DMD]) /: List(secondToLast, rest)) { case ((zAccum, aAccum), (zR, aR)) => ((zAccum :+ zR), aR :: aAccum/*(aAccum :+ aR)*/) }
			


			val lastLayer = layers.last

			//val deltaForFinalLayer = costFunc.matrixDelta(a, oMatrix, z, lastLayer.fPrime)
			val deltaForFinalLayer = costFunc.matrixDelta(last, oMatrix)
			val gradForFinalLayer = lastLayer match {
				case gb : GradBuilder[_] => Some(gb.buildGrad(deltaForFinalLayer, secondToLast))
				case _ => None
			}
			println("deltaForFinalLayer")

		  //val deltas = calcDeltas(weights, List(deltaForFinalLayer), restZ)
		  val grads = calcGrads(layers, List(deltaForFinalLayer), secondToLast :: rest, List[Option[Grad[_]]](gradForFinalLayer))
		  println("deltas")

		  /*val dAndA = deltas.zip(restA)

		  val (gradWs, gradBs) = (((List.empty[DMD], List.empty[DVD]) /: dAndA) {
		  	case ((ws, bs), (d, prevA)) => ((d * prevA.t) :: ws, sum(d(*, ::)) :: bs)
		  })

		  println("gradW grad B")

		  return (gradWs.reverse, gradBs.reverse)*/
		  return grads
		  //return dAndA.foldLeft(Network.empty(f, fPrime)) {
		  	//case (Network(ws, bs, f, fP), (d : DMD, prevA : DMD)) => //Network(ws :+ (d * prevA.t), bs :+ sum(d(*, ::)), f, fP)
		  //}
		}
	}

	case class TrainingExample(input : DVD, output : DVD)

	type TrainingBatch = scala.collection.immutable.Vector[TrainingExample]

	def sigmoid(z : Double) : Double = 1 / (1 + Math.exp(-z))

	def sigmoidPrime(z : Double) : Double = sigmoid(z) * (1 - sigmoid(z))

	//def meanSquareError(a : Double, y : Double) : Double = Math.pow(y - a, 2) /2

	def meanSquareErrorPrime(a : Double, y : Double) : Double = a - y

	//def crossEntropy(a : Double, y : Double) : Double =  -(y * Math.log(a) + ((1 - y) * Math.log(1 - a)))

	def crossEntropyPrime(a : Double, y : Double) : Double = (a -y) / (a * (1 - a))

	def listToColumnMatrix(lst : List[Double]) : DMD = new DenseMatrix(lst.length, 1, lst.toArray)

	def listToDVD(lst : List[Double]) : DVD = new DenseVector(lst.toArray)

	def batchToIOMatrices(b : TrainingBatch) : (DMD, DMD) = {
		if (b.length == 0)
		{
			return (DenseMatrix.zeros[Double](0, 0), DenseMatrix.zeros[Double](0, 0))
		}

		val iRows = b(0).input.length;
		val oRows = b(0).output.length;

		if (iRows == 0 || oRows == 0)
		{
			return (DenseMatrix.zeros[Double](0, b.length), DenseMatrix.zeros[Double](0, b.length))
		}

		val cols = b.length;

		val (inputMatrix, outputMatrix) = ( (b.head.input.asDenseMatrix.t, b.head.output.asDenseMatrix.t) /: b.tail) {
			case ((iAccum, oAccum), TrainingExample(input, output)) => (DenseMatrix.horzcat(iAccum, input.asDenseMatrix.t), DenseMatrix.horzcat(oAccum, output.asDenseMatrix.t))
		}

		return (inputMatrix, outputMatrix)
	}

	def MNISTLabelToDVD(l : Int) = new DenseVector(
		for {
			x <- (0 until 10).toArray;
			y = if(x == l) 1.0 else 0.0
		} yield y) //timeThis("MNISTLabelToList"){(0 to 9).map(n => if (n == l) 1.0 else 0.0).toList}

	def MNISTImageToList(img : MNISTImage) = new DenseVector(img.data.map(_.toDouble / 255.0))

	def getMNISTData : Option[TrainingData] = {
		println("Starting getMNISTData Operation")
		def getResourcePath(path : String) = getClass().getResource(path).getPath()

		val trainingImagesPath = "/train-images-idx3-ubyte"
		val trainingLabelsPath = "/train-labels-idx1-ubyte"
		val testImagesPath = "/t10k-images-idx3-ubyte"
		val testLabelsPath = "/t10k-labels-idx1-ubyte"

		println("Reading MNIST data...")
		val trainingImages = timeThis("loadTrainingImages"){MNISTLoader.readImages(getResourcePath(trainingImagesPath))}
		val trainingLabels = timeThis("loadTrainingLabels"){MNISTLoader.readLabels(getResourcePath(trainingLabelsPath))}
		val testImages = MNISTLoader.readImages(getResourcePath(testImagesPath))
		val testLabels = MNISTLoader.readLabels(getResourcePath(testLabelsPath))
		println("Reading complete. Preparing...")

		(trainingImages, trainingLabels, testImages, testLabels) match {
			case (Success(trI), Success(trL), Success(teI), Success(teL)) => {
				val s1 =
					(timeThis("trI map"){trI map(MNISTImageToList)}
						zip(
							trL.map(MNISTLabelToDVD)
						)) map(t => TrainingExample(t._1, t._2))
				val s2 =
					teI.map(MNISTImageToList)
						.zip(
							teL.map(MNISTLabelToDVD)
						).map(t => TrainingExample(t._1, t._2))
				println("Finished preparing MNIST Data. Operation Complete.")
				//println(s1(0).input)
				return Option(TrainingData(s1.take(50000), s2, s1.drop(50000)))
			}
			case _ => {
				println(s"getMNISTData Failed: $trainingImages $trainingLabels $testImages $testLabels");
				return None
			}
		}
	}

	def main(args: Array[String]): Unit = {

		//val n = Network(List(new DenseMatrix(1, 2, Array(1.0, 2.0))), List(DenseVector(1.0)), identity, (x => 1));
		//println(s"feedforward basic test: ${n.feedforward(List(1.0, 1.0))}")
		println(DenseMatrix((1.0, 2.0), (2.0, 3.0)) * DenseMatrix((1.0, 2.0), (2.0, 3.0)))
		val td = getMNISTData
		//val n = Network.withLayerSizes(List(28 * 28, 30, 10), sigmoid, sigmoidPrime)
		def fclweightInit(size : Int, numOfInputs : Int) = DenseMatrix.rand(size, numOfInputs, Rand.gaussian).map(_ / math.sqrt(numOfInputs.toDouble))
		def convWeightInit(lrfDimensions : Seq[Int], numOfFeatureMaps : Int) : DMD = DenseMatrix.rand[Double](numOfFeatureMaps, lrfDimensions.reduce(_ * _)).map(_ / math.sqrt(lrfDimensions.reduce(_ * _)))
		val n =
			Network(
				List(
					ConvolutionalLayer.gen(
						lrfDimensions = List(5, 5),
						inputDimensions = List(28, 28),
						numOfFeatureMaps = 20,
						f = sigmoid,
						fPrime = sigmoidPrime,
						weightInit = convWeightInit
					),
					MaxPoolLayer.gen(
						lrfDimensions = List(2, 2, 1),
						inputDimensions = List(24,24,20)
					),
					FullyConnectedLayer.gen(
						size = 10,
						numOfInputs = 12 * 12 * 20,
						f = sigmoid,
						fPrime = sigmoidPrime,
						weightInit = fclweightInit
					)
				)
			)
		val t = new Trainer(crossEntropy) with DefaultReporter with DefaultAssessor[CostFuncs.CrossEntropyCostFunc]
		val hp = TrainingParameters(1, 30, 0.5, momentum = 0.0, lmbda = 0.0)
		td.map (data => t.train(n, data, hp)).map(f => concurrent.Await.result(f, concurrent.duration.Duration.Inf))
		Thread.sleep(2000)
		/*val n = Network.withLayerSizes(List(2,1), sigmoid, sigmoidPrime)

		val t = new Trainer(crossEntropyCostFunc)

		val examples = List(
			TrainingExample(List(0.0, 0.0), List(0.0)),
			TrainingExample(List(1.0, 0.0), List(0.0)),
			TrainingExample(List(0.0, 1.0), List(0.0)),
			TrainingExample(List(1.0, 1.0), List(1.0)))

		val tn = t.train(n, examples, 1, 5000, 0.3, earlyStopCount = 50)
		val act = tn.feedforward(List(0.0, 0.0))(0,0)

		println(act)
		println(tn.feedforward(List(1.0, 0.0)))
		println(tn.feedforward(List(0.0, 1.0)))
		println(tn.feedforward(List(1.0, 1.0)))

		println(tn.feedforwardBatch(examples))

		println("Now approximating XOR function")

		val examples2 = List(
			TrainingExample(List(0.0, 0.0), List(1.0)),
			TrainingExample(List(1.0, 0.0), List(0.0)),
			TrainingExample(List(0.0, 1.0), List(0.0)),
			TrainingExample(List(1.0, 1.0), List(1.0)))

		val numberOfDuplicates = 100;
		val noisyExamples = examples2 flatMap(example => example :: (0 until numberOfDuplicates).map ({ x =>
					TrainingExample(example.input.map(_ + 0.15 * (Random.nextDouble - 0.5)), example.output)
		}).toList)

		//println(noisyExamples)

		val n2 = Network.withLayerSizes(List(2, 2, 1), sigmoid, sigmoidPrime)
		//val n2 = Network(List(DenseMatrix((-0.027803368493914613, 0.042694670101627696), (-0.023333012731745845, -0.029483888670802122)), DenseMatrix((0.04057354652322828, 0.012286039395257825))), List(DenseMatrix((0.05699043697677553), (-0.007286587450653317)), DenseMatrix((0.07750239833258094))), sigmoid, sigmoidPrime)

		val tn2 = t.train(n2, noisyExamples, 16, 50000, 0.3, momentum = 0.8, earlyStopCount = 100)
		println(tn2)
		println(s"feed forward test ${tn2.feedforward(List(1.0, 1.0))}")
		println(s"feed forward test ${tn2.feedforward(List(1.0, 0.0))}")*/

	}
}
