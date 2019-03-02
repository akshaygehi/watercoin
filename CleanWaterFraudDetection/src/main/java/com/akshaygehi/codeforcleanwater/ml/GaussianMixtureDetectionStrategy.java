/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;

/**
 * @author akshaygehi
 *
 */
public class GaussianMixtureDetectionStrategy implements FraudDetectionStrategy {

	private static final int MAX_ITERATIONS = 10;

	private static final int NUMBER_OF_CLUSTERS_K = 2;

	private GaussianMixtureModel gmm;
	
	private static final String MODEL_LOCATION = "models/GaussianMixtureModel.mdl";

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#trainModel(org.apache.spark.api.java.JavaRDD)
	 */
	@Override
	public void trainModel(JavaRDD<Vector> data) {
		gmm = new GaussianMixture().setK(NUMBER_OF_CLUSTERS_K)
				.setMaxIterations(MAX_ITERATIONS).run(data.rdd());
//		// Output the parameters of the mixture model
//		for (int j = 0; j < gmm.k(); j++) {
//		  System.out.printf("weight=%f\nmu=%s\nsigma=\n%s\n",
//		    gmm.weights()[j], gmm.gaussians()[j].mu(), gmm.gaussians()[j].sigma());
//		}
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#saveModel()
	 */
	@Override
	public void saveModel() {
		gmm.save(SparkSupport.sc.sc(), MODEL_LOCATION);
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#loadModel()
	 */
	@Override
	public void loadModel() {
		JavaSparkContext sc = SparkSupport.sc;
		gmm = GaussianMixtureModel.load(sc.sc(), MODEL_LOCATION);
	}

}
