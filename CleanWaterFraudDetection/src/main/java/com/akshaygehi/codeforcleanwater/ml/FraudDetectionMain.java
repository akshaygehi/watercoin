package com.akshaygehi.codeforcleanwater.ml;

import java.io.File;
import java.net.MalformedURLException;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

public class FraudDetectionMain {
	
	private static final String SAMPLE = "/Users/akshaygehi/Spark/CleanWaterFraudDetection/src/main/resources/LrgInd/IND100B.csv";
	
	public static void main(String[] args) throws Exception {
		VectorParser parser = new VectorParser();
		JavaSparkContext sc = SparkSupport.sc;
		
		// Easily use Streams instead
		JavaRDD<String> input = parseFile(sc, SAMPLE);
		
		// Convert this into vector so we can use various algorithms on the same data
		JavaRDD<Vector> inputVectors = parser.parseData(input);
		inputVectors.cache();
		
		FraudDetectionStrategy strategy1 = new KmeansFraudDetectionStrategy();
		strategy1.trainModel(inputVectors);
		
		FraudDetectionStrategy strategy2 = new GaussianMixtureDetectionStrategy();
		strategy2.trainModel(inputVectors);
		
	}
	
	private static JavaRDD<String> parseFile(JavaSparkContext sc, String sampleFile) throws MalformedURLException {
		JavaRDD<String> data = sc.textFile(new File(sampleFile).toURI().toURL().toString(), 4);
		return data;
	}
}
