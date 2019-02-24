/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

/**
 * @author akshaygehi
 *
 */
public class KmeansFraudDetectionStrategy implements FraudDetectionStrategy {

	int numClusters = 3;
	int numIterations = 5;
	
	KMeansModel model;
	private double cost;
	
	@Override
	public void trainModel(JavaRDD<Vector> data) {
		// Cluster the data into two classes using Kmeans
		model = KMeans.train(data.rdd(), numClusters, numIterations);
		
		System.out.println("Cluster centers:");
		for (Vector center: model.clusterCenters()) {
		  System.out.println(" " + center);
		}
		
		cost = model.computeCost(data.rdd());
		System.out.println("Cost: " + cost);

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double WSSSE = model.computeCost(data.rdd());
		System.out.println("Within Set Sum of Squared Errors = " + WSSSE);
		
	}

	@Override
	public void saveModel() {
		model.save(SparkSupport.sc.sc(), "target/org/apache/spark/JavaKMeansExample/KMeansModel");
	}

	@Override
	public void loadModel() {
		model = KMeansModel.load(SparkSupport.sc.sc(),
				  "target/org/apache/spark/JavaKMeansExample/KMeansModel");
	}

}
