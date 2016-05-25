package com.example

import breeze.linalg._

import Hello.Network
import Hello.TrainingExample
import Hello.TrainingBatch
import Hello.{DVD, timeThis}

case class NetworkAssessment(correctSet : TrainingBatch, incorrectSet : TrainingBatch, cost : Double)

trait Assessor[A <: CostFunc] {
	val costFunc : A
	def isCorrect(actual : DVD, expected : DVD) : Boolean

	final def assess(n : Network, data : TrainingBatch) : NetworkAssessment = {
		val start = (scala.collection.immutable.Vector.empty[TrainingExample], scala.collection.immutable.Vector.empty[TrainingExample], 0.0)

		(start /: (data)) { case ((corrects, incorrects, cost), example) =>
					val expected = example.output
					val actual = n.feedforward(example.input)

					val cf = actual.mapPairs ({ case ((r, _), a) => costFunc(a, expected(r)) })
					val newCost = cost + norm(cf(::, 0))

					if (isCorrect(actual(::, 0), expected))
						(corrects :+ example, incorrects, newCost)
					else
						(corrects, incorrects :+ example, newCost)
		} match {
			case (c, i, cost) => NetworkAssessment(c, i, cost)
		}
	}
}

trait DefaultAssessor[A <: CostFunc] extends Assessor[A] {
	def isCorrect(actual : DVD, expected : DVD) : Boolean = argmax(actual) == argmax(expected)
}