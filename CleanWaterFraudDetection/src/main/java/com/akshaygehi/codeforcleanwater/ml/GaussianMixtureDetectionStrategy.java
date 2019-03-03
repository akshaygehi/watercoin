/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

/**
 * @author akshaygehi
 *
 */
public class GaussianMixtureDetectionStrategy implements FraudDetectionStrategy {

	private static final int MAX_ITERATIONS = 10;
	private static final int NUMBER_OF_CLUSTERS_K = 2;
	
	private static final String MODEL_LOCATION = "models/GaussianMixtureModel";
	
	private GaussianMixtureModel gmm;

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#trainModel(org.apache.spark.api.java.JavaRDD)
	 */
	@Override
	public void trainModel(Dataset<Row> data) {
		GaussianMixture gm = new GaussianMixture()
				.setK(NUMBER_OF_CLUSTERS_K)
				.setMaxIter(MAX_ITERATIONS);
		gmm = gm.fit(data);
		
		// Output the parameters of the mixture model
		for (int j = 0; j < gmm.getK(); j++) {
			System.out.printf("weight=%f\nmean=%s\n", gmm.weights()[j], gmm.gaussians()[j].mean());
		}
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#saveModel()
	 */
	@Override
	public void saveModel() {
//		gmm.save(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#loadModel()
	 */
	@Override
	public void loadModel() {
		JavaSparkContext sc = SparkSupport.sc;
//		gmm = GaussianMixtureModel.load(sc.sc(), MODEL_LOCATION);
	}

	@Override
	public PipelineStage getStage() {
		GaussianMixture gm = new GaussianMixture()
				.setK(NUMBER_OF_CLUSTERS_K)
				.setMaxIter(MAX_ITERATIONS);
		
		return gm;
	}

}
