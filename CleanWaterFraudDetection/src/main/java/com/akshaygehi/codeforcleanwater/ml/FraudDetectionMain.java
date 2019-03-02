package com.akshaygehi.codeforcleanwater.ml;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.MalformedURLException;
import java.net.URL;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;

public class FraudDetectionMain {
	
	private static final String SAMPLE_FILE = "LrgInd/IND100B.csv";
	
	public static void main(String[] args) throws Exception {
		VectorParser parser = new VectorParser();
		JavaSparkContext sc = SparkSupport.sc;
		
		String file = locateSampleFile();
		
		// Easily use Streams instead
		JavaRDD<String> input = parseFile(sc, file);
		
		// Convert this into vector so we can use various algorithms on the same data
		JavaRDD<Vector> inputVectors = parser.parseData(input);
		inputVectors.cache();
		
		FraudDetectionStrategy strategy1 = new KmeansFraudDetectionStrategy();
		strategy1.trainModel(inputVectors);
		
		FraudDetectionStrategy strategy2 = new GaussianMixtureDetectionStrategy();
		strategy2.trainModel(inputVectors);
		
	}

	private static String locateSampleFile() throws FileNotFoundException {
		URL fileUrl = ClassLoader.getSystemResource(SAMPLE_FILE);
		if(fileUrl == null) {
			throw new FileNotFoundException("The following file could not be located on the classpath: " + SAMPLE_FILE);
		}
		
		String file = fileUrl.getFile();
		return file;
	}
	
	private static JavaRDD<String> parseFile(JavaSparkContext sc, String sampleFile) throws MalformedURLException {
		JavaRDD<String> data = sc.textFile(new File(sampleFile).toURI().toURL().toString(), 4);
		return data;
	}
}
