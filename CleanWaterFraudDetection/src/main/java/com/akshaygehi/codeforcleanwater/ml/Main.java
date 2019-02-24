package com.akshaygehi.codeforcleanwater.ml;

import java.io.File;
import java.net.MalformedURLException;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.GaussianMixture;
import org.apache.spark.mllib.clustering.GaussianMixtureModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Main {
	
	private static final String SAMPLE = "/Users/akshaygehi/Spark/CleanWaterFraudDetection/src/main/resources/LrgInd/IND100A.csv";

	public static void main(String[] args) throws MalformedURLException {
		JavaSparkContext sc = SparkSupport.sc;
		
		JavaRDD<String> data = sc.textFile(new File(SAMPLE).toURI().toURL().toString(), 4);
		
		JavaRDD<Vector> parsedData = data.filter(l -> !l.startsWith("id")).map((String l) -> {
			String[] split = l.split(",");
			split[0] = split[0].replace("IND", "0");
			split[0] = split[0].replace("A", "0");
			
			double[] values = new double[split.length];
			  for (int i = 0; i < split.length; i++) {
			    values[i] = Double.parseDouble(split[i]);
			  }
			  
			  return Vectors.dense(values);
		});
		
		parsedData.cache();
		
		// Cluster the data into two classes using GaussianMixture
		GaussianMixtureModel gmm = new GaussianMixture().setK(2)
				.setMaxIterations(10).run(parsedData.rdd());
		
		// Save and load GaussianMixtureModel
		
//		gmm.save(sc.sc(), "target/org/apache/spark/JavaGaussianMixtureExample/GaussianMixtureModel");
//		GaussianMixtureModel sameModel = GaussianMixtureModel.load(sc.sc(),
//		  "target/org.apache.spark.JavaGaussianMixtureExample/GaussianMixtureModel");

		// Output the parameters of the mixture model
		for (int j = 0; j < gmm.k(); j++) {
		  System.out.printf("weight=%f\nmu=%s\nsigma=\n%s\n",
		    gmm.weights()[j], gmm.gaussians()[j].mu(), gmm.gaussians()[j].sigma());
		}
		
	}
	
//	public List<ResultsSummary> executeAlgorithm(JavaSparkContext sparkContext, JavaRDD<String> data) throws IOException {
//
//		for (int i = 0; i < 4; i++) {
//			long startTime = System.currentTimeMillis();
//
//		}
//		return resultsSummaries;
//	}
	
}
