package com.example

import scala.util.{ Success, Try }

import java.io.{ BufferedInputStream, FileInputStream }
import java.nio.channels.FileChannel

import Hello.timeThis

case class MNISTImage(rows : Int, cols : Int, data : Array[Int])

object MNISTLoader {

	def to32bitInt(buf : Array[Byte]) = (0 until 4).map(i => (buf(i) & 0xFF) << ((3 - i) * 8)).reduce(_ + _)

	def readLabels(path : String) : Try[Vector[Int]] = Try {
		val bufSize = 1024
		val buf = new Array[Byte](bufSize)

		val gis = new BufferedInputStream(new FileInputStream(path))
		gis.read(buf, 0, 4)
		val magicNumber = to32bitInt(buf)
		val expectedMN = 2049

		assert(magicNumber == expectedMN)

		gis.read(buf, 0, 4)
		val numberOfImages = to32bitInt(buf)

		val labels = Stream.continually {
			val n = gis.read(buf, 0, bufSize)
			(buf.take(n), n)
			}.takeWhile { case (_, n) => n != -1 }
			.map(_._1)
			.flatMap(_.map((b : Byte) => (b & 0xFF)))
			.toVector

		gis.close()
		return Success(labels)
	}

	def readImages(path : String) : Try[Vector[MNISTImage]] = Try {
		val bufSize = 4096000
		val buf = new Array[Byte](bufSize)

		//TODO:Rewrite to support files larger than 2GB
		val channel = new FileInputStream(path).getChannel()
		val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())

		buffer.get(buf, 0, 4)
		val magicNumber = to32bitInt(buf)
		val expectedMN = 2051

		assert(magicNumber == expectedMN)

		buffer.get(buf, 0, 4)
		val numberOfImages = to32bitInt(buf)

		buffer.get(buf, 0, 4)
		val rows = to32bitInt(buf)

		buffer.get(buf, 0, 4)
		val cols = to32bitInt(buf)

		val imgSize = rows * cols;

		val pixelBytes = timeThis("reading"){
			val ba = new Array[Byte](buffer.remaining())
			buffer.get(ba)
			ba
		}
		val images = ((0 until pixelBytes.length by imgSize) map { i =>
						val pixels = pixelBytes.slice(i, Math.min(i + imgSize, pixelBytes.length)).map(b => if(b.toInt < 0) 256 + b else b.toInt)
						MNISTImage(rows, cols, pixels)
			}).toVector

		channel.close()
		return Success(images)
	}
}