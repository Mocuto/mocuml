package com.example

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.{Success, Failure}
import Hello.{ Network, TrainingData, TrainingParameters, TrainingResult }

case class DiagnosticsResponse(msg : String)

trait Reporter {
	final def whenTrainingStarts(n : Network, td : TrainingData, hp : TrainingParameters)(trainingFut : Future[TrainingResult]) : Future[TrainingResult] = {
		val onTrainingCompleted = onTrainingStarted(n, trainingFut)

		trainingFut.onComplete {
			case Success(trainingResult) => onTrainingCompleted(trainingResult, td)
			case Failure(t) => handleTrainingException(t)
		}

		return trainingFut
	}

	final protected def whenEpochStarts(n : Network, td : TrainingData, hp : TrainingParameters, epochsLeft : Int)(func : => (Network, Network)) : (Network, Network) = {
		val onEpochCompleted = onEpochStarted(n, td, hp, epochsLeft)
		val (newN, v) = func

		Future {
			onEpochCompleted(newN, td, hp)
		}

		return (newN, v)
	}

	def onTrainingStarted(n : Network, f : Future[TrainingResult]) : ((TrainingResult, TrainingData) => Unit)
	def onEpochStarted(n : Network, td : TrainingData, hp : TrainingParameters, epochsLeft : Int) : ((Network, TrainingData, TrainingParameters) => Unit)

	def handleTrainingException(t : Throwable) : Unit = println("An error occured training the network:" + t)

}

trait DefaultReporter extends Reporter {
	def onTrainingStarted(n : Network, training : Future[TrainingResult]) : ((TrainingResult, TrainingData) => Unit) = {
		println("Neural Network Training has Started")
		println()
		val startTime = System.currentTimeMillis

		return (tr, _) => {
			val endTime = System.currentTimeMillis()
			println(s"Training has finished after ${(endTime - startTime) / 1000} second(s)")
			println(s"exampleAssessment: ${tr.exampleAssessment}")
			println(s"testAssessment: ${tr.testAssessment}")
			println(s"validationAssessment: ${tr.validationAssessment}")
		}
	}

	def onEpochStarted(n : Network, td : TrainingData, hp : TrainingParameters, epochsLeft : Int) : ((Network, TrainingData, TrainingParameters) => Unit) = {
		val startTime = System.currentTimeMillis()
		val epochNum = hp.epochs - epochsLeft

		val a = new DefaultAssessor[CostFuncs.CrossEntropyCostFunc] {
			val costFunc = CostFuncs.crossEntropy
		}

		return (newNet, td, _) => {
			val endTime = System.currentTimeMillis()
			println(s"Epoch $epochNum finished in ${(endTime - startTime) / 1000} second(s)")
			val NetworkAssessment(correct, incorrect, _) = a.assess(newNet, td.testSet)
			println(s"Epoch $epochNum accuracy: ${correct.length} / ${correct.length + incorrect.length}")
		}
	}
}