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

	private static final int MAX_ITERATIONS = 10;
	private static final int NUMBER_OF_CLUSTERS_K = 2;
	
	private static final String MODEL_LOCATION = "models/KMeansModel";
	
	KMeansModel model;
	private double cost;
	
	@Override
	public void trainModel(JavaRDD<Vector> data) {
		// Cluster the data into two classes using Kmeans
		model = KMeans.train(data.rdd(), NUMBER_OF_CLUSTERS_K, MAX_ITERATIONS);
		
		System.out.println("Cluster centers:");
		for (Vector center: model.clusterCenters()) {
		  System.out.println(" " + center);
		}
		
		cost = model.computeCost(data.rdd());
		System.out.println("Cost: " + cost);

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double wsse = model.computeCost(data.rdd());
		System.out.println("Within Set Sum of Squared Errors = " + wsse);
		
	}

	@Override
	public void saveModel() {
		model.save(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

	@Override
	public void loadModel() {
		model = KMeansModel.load(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

}
