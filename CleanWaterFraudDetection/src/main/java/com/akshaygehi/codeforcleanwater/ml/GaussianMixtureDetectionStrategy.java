/**
 * 
 */
package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;

/**
 * @author akshaygehi
 *
 */
public class GaussianMixtureDetectionStrategy implements FraudDetectionStrategy {

	private GaussianMixtureModel gmm;

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#trainModel(org.apache.spark.api.java.JavaRDD)
	 */
	@Override
	public void trainModel(JavaRDD<Vector> data) {
		gmm = new GaussianMixture().setK(2)
				.setMaxIterations(10).run(data.rdd());
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
		gmm.save(SparkSupport.sc.sc(), "target/org/apache/spark/JavaGaussianMixtureExample/GaussianMixtureModel");
	}

	/* (non-Javadoc)
	 * @see com.akshaygehi.codeforcleanwater.ml.FraudDetectionStrategy#loadModel()
	 */
	@Override
	public void loadModel() {
		gmm = GaussianMixtureModel.load(SparkSupport.sc.sc(),
				  "target/org/apache/spark/JavaGaussianMixtureExample/GaussianMixtureModel");
	}

}
