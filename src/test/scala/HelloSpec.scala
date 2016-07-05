import org.scalatest._
import com.example._
import com.example.Hello._
import breeze.linalg._

class HelloSpec extends FlatSpec with Matchers {
  "Hello" should "have tests" in {
    true should === (true)
  }

  "ConvFeedFordward" should "format input data correctly" in {
  	val l = ConvolutionalLayer.gen(List(2,2), List(3, 3), identity, identity, 1)
  	val i = DenseMatrix(
  		(1.0, 2.0, 3.0),
  		(4.0, 5.0, 6.0),
  		(7.0, 8.0, 9.0))

  	val expected = DenseVector(
  		1.0, 2.0, 4.0, 5.0,
  		4.0, 5.0, 7.0, 8.0,
  		2.0, 3.0, 5.0, 6.0,
  		5.0, 6.0, 8.0, 9.0).asDenseMatrix.t.reshape(4, 4)

  	val formmatted = l.format(i)
  	formmatted should equal (expected)
  }

  it should "unformat data correctly" in {
  	val l = ConvolutionalLayer.gen(List(2,2), List(3, 3), identity, identity, 1)

  	val formmatted = DenseVector(
  		1.0, 2.0, 4.0, 5.0,
  		4.0, 5.0, 7.0, 8.0,
  		2.0, 3.0, 5.0, 6.0,
  		5.0, 6.0, 8.0, 9.0).asDenseMatrix.t.reshape(4, 4)

  	val expected = DenseVector(
  		1.0, 2.0, 4.0, 5.0,
  		4.0, 5.0, 7.0, 8.0,
  		2.0, 3.0, 5.0, 6.0,
  		5.0, 6.0, 8.0, 9.0).asDenseMatrix.t

  	val unformatted = l.unformat(formmatted)

  	unformatted should equal (expected)
  }

  it should "feedforward correctly" in {
  	val l = ConvolutionalLayer(
  		weights = DenseVector(1.0, 2.0, 3.0, 4.0).asDenseMatrix,
  		biases = DenseVector(1.0),
  		f = identity,
  		fPrime = identity,
  		lrfDimensions = List(2,2),
  		inputDimensions = List(3,3),
  		featureMaps = 1
  	)

  	val expected = DenseVector(36.0, 46.0 , 66.0, 76.0).asDenseMatrix
  	val input = DenseMatrix((1.0,2.0,3.0),(4.0,5.0,6.0),(7.0, 8.0, 9.0))
  	val (actualZ, actualA) = l.feedforward(input)

  	actualZ should equal (expected)
  }

  it should "have the right hidden layer dimensions" in {
  	val l = ConvolutionalLayer(
  		weights = DenseVector(1.0, 2.0, 3.0, 4.0).asDenseMatrix,
  		biases = DenseVector(1.0),
  		f = identity,
  		fPrime = identity,
  		lrfDimensions = List(2,2),
  		inputDimensions = List(3,3),
  		featureMaps = 1
  	)
	val expected = List(2,2)
	val actual = l.hiddenDimensions.toList

	actual should equal (expected)
  }

  "The new delta method in layer" should "return the same value as the original Network.backprop" in {
  	val n = Network.withLayerSizes(List(2,2, 1), sigmoid, sigmoidPrime)

 		val examples = List(
			TrainingExample(DenseVector(0.0, 0.0), DenseVector(0.0)),
			TrainingExample(DenseVector(1.0, 0.0), DenseVector(0.0)),
			TrainingExample(DenseVector(0.0, 1.0), DenseVector(0.0)),
			TrainingExample(DenseVector(1.0, 1.0), DenseVector(1.0)))

 		for(e <- examples)
 		{
 			val (_, originalDeltas) = n.backprop(scala.collection.immutable.Vector(e), CostFuncs.meanSquareError)
 			val (outputs, activations) = n.feedforwardAccum(0)(List((e.input.asDenseMatrix.t, e.input.asDenseMatrix.t))).unzip

 			val deltaForFinalLayer = CostFuncs.meanSquareError.matrixDelta(activations.head, e.output.asDenseMatrix.t, outputs.head, n.layers.last.fPrime)
 			println(s"deltaForFinalLayer $deltaForFinalLayer activations.head ${activations.head} e.output ${e.output} outputs.ast ${outputs.head}")

 			val newDeltas = (List(deltaForFinalLayer) /: (n.layers.reverse.sliding(2).toList.zip(outputs.reverse.tail))) { case (deltas, (prevLayer :: (layer :: _), z)) =>
 				val (delta :: _) = deltas
 				val nextDelta = layer.delta(z, delta, prevLayer.weights)

 				nextDelta :: deltas
 			}

 			for((oD, nD) <- originalDeltas.zip(newDeltas))
 			{
 				nD.toDenseVector should equal (oD)
 			}
 		}
  }

  /*"A net with a FCL -> CVL" should "backpropogate correctly" in {
  	val fclWeightInit = (size : Int, numOfInputs : Int) => DenseMatrix.ones[Double](size, numOfInputs)
  	val fclBiasInit = (size : Int) => DenseVector.ones[Double](size)
  	val fcl = FullyConnectedLayer.gen(9, 9, identity, (d : Double) => 1.0, weightInit = fclWeightInit, biasInit = fclBiasInit)

  	val cvlWeightInit = (lrfDimensions : Seq[Int], numOfFeatureMaps : Int) => DenseMatrix.ones[Double](numOfFeatureMaps, lrfDimensions.reduce(_ * _))
  	val cvlBiasInit = (numOfFeatureMaps : Int) => DenseVector.ones[Double](numOfFeatureMaps)
  	val cvl = ConvolutionalLayer.gen(List(2,2), List(3,3), identity, (d : Double) => 1.0, 1, weightInit = cvlWeightInit, biasInit = cvlBiasInit)

  	val input = DenseVector(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0).asDenseMatrix.t
    val output = DenseVector(34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0, 34.0).asDenseMatrix.t

  	val (z0, a0) = fcl.feedforward(input)
  	val (z1, a1) = cvl.feedforward(a0)

    val expectedDelta2 = DenseVector(
      17.0, 17.0,
      17.0, 17.0)

    val expectedDelta1 - DenseVector(17.0, 34.0, 51.0, 17.0, 34.0, 51.0, 17.0, 34.0, 51.0)

    val n = Network(List(fcl, cvl))

    val (_, deltas) = n.backprop(List(TrainingExample(input, output)), CostFuncs.meanSquareError)
  }*/
}
