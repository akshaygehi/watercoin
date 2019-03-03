/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * @author akshaygehi
 *
 */
public class KmeansFraudDetectionStrategy implements FraudDetectionStrategy {

	private static final int MAX_ITERATIONS = 10;
	private static final int NUMBER_OF_CLUSTERS_K = 2;

	private static final String MODEL_LOCATION = "models/KMeansModel";

	KMeansModel model;

	@Override
	public void trainModel(Dataset<Row> data) {
		
		// // Cluster the data into two classes using Kmeans
		KMeans kmeans = new KMeans().setK(NUMBER_OF_CLUSTERS_K).setMaxIter(MAX_ITERATIONS);
		model = kmeans.fit(data);

		System.out.println("Cluster centers:");

		for (Vector center : model.clusterCenters()) {
			System.out.println(" " + center);
		}

		// Evaluate clustering by computing Within Set Sum of Squared Errors
		double wsse = model.summary().trainingCost();
		System.out.println("Within Set Sum of Squared Errors = " + wsse);

	}

	@Override
	public void saveModel() {
		// model.save(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

	@Override
	public void loadModel() {
		// model = KMeansModel.load(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

	@Override
	public PipelineStage getStage() {
		// Cluster the data into two classes using Kmeans
		KMeans kmeans = new KMeans().setK(NUMBER_OF_CLUSTERS_K).setMaxIter(MAX_ITERATIONS);
		return kmeans;
	}

}
