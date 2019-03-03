/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import java.io.IOException;

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
		// Save data.
		try {
			gmm.write().save(MODEL_LOCATION);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#loadModel()
	 */
	@Override
	public void loadModel() {
		// Loads data.
		Dataset<Row> dataset = SparkSupport.sparkSession.read().format("libsvm").load(MODEL_LOCATION);
		gmm = getStage().fit(dataset);
	}

	@Override
	public GaussianMixture getStage() {
		GaussianMixture gm = new GaussianMixture()
				.setK(NUMBER_OF_CLUSTERS_K)
				.setMaxIter(MAX_ITERATIONS);
		
		return gm;
	}

	@Override
	public Dataset<Row> transform(Dataset<Row> data) {
		return gmm.transform(data);
	}

}
