/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.optimization

import scala.collection.mutable.ArrayBuffer
import breeze.linalg.{norm, DenseVector => BDV}
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}

import scala.collection.mutable.Queue
import scala.concurrent.Await
import scala.concurrent.duration.Duration

/**
  * Class used to solve an optimization problem using Gradient Descent.
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
//class GradientDescent private[spark] (private var gradient: Gradient, private var updater: Updater)
//  extends Optimizer with Logging {
//
//  private var stepSize: Double = 1.0
//  private var numIterations: Int = 100
//  private var regParam: Double = 0.0
//  private var miniBatchFraction: Double = 1.0
//  private var convergenceTol: Double = 0.001
//
//  /**
//    * Set the initial step size of SGD for the first step. Default 1.0.
//    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
//    */
//  def setStepSize(step: Double): this.type = {
//    this.stepSize = step
//    this
//  }
//
//  /**
//    * :: Experimental ::
//    * Set fraction of data to be used for each SGD iteration.
//    * Default 1.0 (corresponding to deterministic/classical gradient descent)
//    */
//  @Experimental
//  def setMiniBatchFraction(fraction: Double): this.type = {
//    this.miniBatchFraction = fraction
//    this
//  }
//
//  /**
//    * Set the number of iterations for SGD. Default 100.
//    */
//  def setNumIterations(iters: Int): this.type = {
//    this.numIterations = iters
//    this
//  }
//
//  /**
//    * Set the regularization parameter. Default 0.0.
//    */
//  def setRegParam(regParam: Double): this.type = {
//    this.regParam = regParam
//    this
//  }
//
//  /**
//    * Set the convergence tolerance. Default 0.001
//    * convergenceTol is a condition which decides iteration termination.
//    * The end of iteration is decided based on below logic.
//    *
//    *  - If the norm of the new solution vector is >1, the diff of solution vectors
//    *    is compared to relative tolerance which means normalizing by the norm of
//    *    the new solution vector.
//    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
//    *    is compared to absolute tolerance which is not normalizing.
//    *
//    * Must be between 0.0 and 1.0 inclusively.
//    */
//  def setConvergenceTol(tolerance: Double): this.type = {
//    require(0.0 <= tolerance && tolerance <= 1.0)
//    this.convergenceTol = tolerance
//    this
//  }
//
//  /**
//    * Set the gradient function (of the loss function of one single data example)
//    * to be used for SGD.
//    */
//  def setGradient(gradient: Gradient): this.type = {
//    this.gradient = gradient
//    this
//  }
//
//
//  /**
//    * Set the updater function to actually perform a gradient step in a given direction.
//    * The updater is responsible to perform the update from the regularization term as well,
//    * and therefore determines what kind or regularization is used, if any.
//    */
//  def setUpdater(updater: Updater): this.type = {
//    this.updater = updater
//    this
//  }
//
//  /**
//    * :: DeveloperApi ::
//    * Runs gradient descent on the given training data.
//    * @param data training data
//    * @param initialWeights initial weights
//    * @return solution vector
//    */
//  @DeveloperApi
//  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
//    val (weights, _) = GradientDescent.runMiniBatchSGD(
//      data,
//      gradient,
//      updater,
//      stepSize,
//      numIterations,
//      regParam,
//      miniBatchFraction,
//      initialWeights,
//      convergenceTol)
//    weights
//  }
//
//}

/**
  * :: DeveloperApi ::
  * Top-level method to run gradient descent.
  */
@DeveloperApi
object GD extends Logging {
  private def isConverged(
                           previousWeights: Vector,
                           currentWeights: Vector,
                           convergenceTol: Double): Boolean = {
    // To compare with convergence tolerance.
    val previousBDV = previousWeights.toBreeze.toDenseVector
    val currentBDV = currentWeights.toBreeze.toDenseVector

    // This represents the difference of updated weights in the iteration.
    val solutionVecDiff: Double = norm(previousBDV - currentBDV)

    solutionVecDiff < convergenceTol * Math.max(norm(currentBDV), 1.0)
  }

  object VectorAccumulatorParam extends AccumulatorParam[BDV[Double]] {
    def zero(initialValue: BDV[Double]): BDV[Double] = {
      BDV.zeros(initialValue.size)
    }
    def addInPlace(v1: BDV[Double], v2: BDV[Double]): BDV[Double] = {
      v1 += v2
    }
  }

  def runMiniBatchSGDBSP(
                          data: RDD[(Double, Vector)],
                          gradient: Gradient,
                          updater: Updater,
                          stepSize: Double,
                          numIterations: Int,
                          regParam: Double,
                          miniBatchFraction: Double,
                          initialWeights: Vector,
                          convergenceTol: Double,
                          statelessIteration: Int): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    /**
      * For the first iteration, the regVal will be initialized as sum of weight squares
      * if it's L2 updater; for L1 updater, the same logic is followed.
      */
    val n = initialWeights.size
    var regVal = updater.compute(
      initialWeights, Vectors.zeros(initialWeights.size), 0, 1, regParam)._2

    //scalastyle:off
    val acc = data.context.accumulator(new BDV[Double](initialWeights.toArray.clone()))(VectorAccumulatorParam)
    var i = 1

    val queue = new Queue[(FutureAction[Unit], RDD[BDV[Double]])]()
    val bcQueue = new Queue[Broadcast[Vector]]()
    var previousRdd : RDD[BDV[Double]] = null
    var currentRdd : RDD[BDV[Double]] = null

    while (i <= numIterations) {
      val bcWeights = data.context.broadcast(Vectors.fromBreeze(acc.value))
      bcQueue.enqueue(bcWeights)

      while(i<= numIterations && queue.size <= statelessIteration) {
        if(i == 1) {
          currentRdd = data.mapPartitions { points =>
            val gradientPerPartition = BDV.zeros[Double](n)
            points.foreach { point =>
              gradient.compute(point._2, point._1, bcWeights.value,
                Vectors.fromBreeze(gradientPerPartition))
            }
            Iterator(gradientPerPartition)
          }.cache()
        } else {
          currentRdd = data.zipPartitionsWithSSPDependency(previousRdd) { (points, rddIter) =>
            val gradientPerPartition = BDV.zeros[Double](n)
            val g = rddIter.next()
            val localWeights = Vectors.fromBreeze(bcWeights.value.toBreeze + (g/numExamples.toDouble)*(-stepSize/math.sqrt(i)))

            points.foreach { point =>
              gradient.compute(point._2, point. _1, localWeights,
                Vectors.fromBreeze(gradientPerPartition))
            }
            Iterator(gradientPerPartition)
          }.cache()
        }
        val job = currentRdd.foreachAsync{x => acc += ((x/numExamples.toDouble)*(-stepSize/math.sqrt(i)))}
        queue.enqueue((job, previousRdd))
        previousRdd = currentRdd
        i += 1
      }
      if(queue.size > statelessIteration) {
        val (job, rdd) = queue.dequeue()
        println("wait for job: " + job.jobIds +"Finished")
        Await.result(job, Duration.Inf)
        if(rdd != null) {
          rdd.unpersist()
        }
      }
//      if(bcQueue.size > 2) { //only need to keep latest 2 broadcast
//        SparkEnv.get.blockManager.removeBroadcast(bcQueue.dequeue.id, true)
//      }
    }

    while(!queue.isEmpty) {
      val (job, rdd) = queue.dequeue()
      Await.result(job, Duration.Inf)
      rdd.unpersist()
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (Vectors.fromBreeze(acc.value), stochasticLossHistory.toArray)
  }

  def runMiniBatchSGD(
                          data: RDD[(Double, Vector)],
                          gradient: Gradient,
                          updater: Updater,
                          stepSize: Double,
                          numIterations: Int,
                          regParam: Double,
                          miniBatchFraction: Double,
                          initialWeights: Vector,
                          convergenceTol: Double): (Vector, Array[Double]) = {

    // convergenceTol should be set with non minibatch settings
    if (miniBatchFraction < 1.0 && convergenceTol > 0.0) {
      logWarning("Testing against a convergenceTol when using miniBatchFraction " +
        "< 1.0 can be unstable because of the stochasticity in sampling.")
    }

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    // Record previous weight and current one to calculate solution vector difference

    var previousWeights: Option[Vector] = None
    var currentWeights: Option[Vector] = None

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size

    /**
      * For the first iteration, the regVal will be initialized as sum of weight squares
      * if it's L2 updater; for L1 updater, the same logic is followed.
      */
    var regVal = updater.compute(
      weights, Vectors.zeros(weights.size), 0, 1, regParam)._2

    //scalastyle:off
    var converged = false // indicates whether converged based on convergenceTol
    var i = 1
    while (!converged && i <= numIterations) {
      val bcWeights = data.context.broadcast(weights)

      def simulateWeights(first: (BDV[Double], Double, Int), second: (BDV[Double], Double, Int))
      : (BDV[Double], Double, Int) = {
        (first._1 + second._2, first._2 + second._2, first._3 + second._3)
      }

      val (gradientSum, lossSum, miniBatchSize) = data.mapPartitions { points =>
        var loss = 0.0
        val gradientPerPartition = BDV.zeros[Double](n)
        var size = 0
        points.foreach { point =>
          loss += gradient.compute(point._2, point. _1, bcWeights.value,
            Vectors.fromBreeze(gradientPerPartition))
          size += 1
        }
        Iterator((gradientPerPartition, loss, size))
      }.reduce(simulateWeights)

      if (miniBatchSize > 0) {
        /**
          * lossSum is computed using the weights from the previous iteration
          * and regVal is the regularization value computed in the previous iteration as well.
          */
        stochasticLossHistory.append(lossSum / miniBatchSize + regVal)
        val update = updater.compute(
          weights, Vectors.fromBreeze(gradientSum / miniBatchSize.toDouble),
          stepSize, i, regParam)
        weights = update._1
        regVal = update._2

        previousWeights = currentWeights
        currentWeights = Some(weights)
        if (previousWeights != None && currentWeights != None) {
          converged = isConverged(previousWeights.get,
            currentWeights.get, convergenceTol)
        }
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
      i += 1
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)

  }
}