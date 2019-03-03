package com.akshaygehi.codeforcleanwater.ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ClassPathUtils;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class FraudDetectionMain {

	private static final String SAMPLE_FILE = "LrgInd";

	static final List<FraudDetectionStrategy> strategies = Arrays.asList(new KmeansFraudDetectionStrategy(),
			new GaussianMixtureDetectionStrategy());

	public static void main(String[] args) throws Exception {
		VectorParser parser = new VectorParser();

		List<String> file = listDirectory("/Users/akshaygehi/git/watercoin/CleanWaterFraudDetection/src/main/resources/LrgInd");

		// Easily use Streams instead
		Dataset<Row> input = parseFile(file);

		// Convert this into vector so we can use various algorithms on the same data
		Dataset<Row> inputVectors = parser.parseData(input);

		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[] { "vol", "ph", "solids", "hardness", "oil", "bod" })
				.setOutputCol("features");

//		Pipeline pipeline = new Pipeline().setStages(
//				new PipelineStage[] { assembler, strategies.get(0).getStage(), 
//						strategies.get(1).getStage() });
//		PipelineModel model = pipeline.fit(inputVectors);
		
		Dataset<Row> inputVectorsForProcessing = assembler.transform(inputVectors);
		inputVectorsForProcessing.cache();
		
		Dataset<Row>[] trainTestData = inputVectorsForProcessing.randomSplit(new double[] { 0.75d, 0.25d });
		
		Dataset<Row> trainData = trainTestData[0];
		Dataset<Row> testData = trainTestData[1];
		
		strategies.forEach(s -> s.trainModel(trainData));

		strategies.forEach(s -> {
			// Evaluate clustering by computing Silhouette score
			ClusteringEvaluator evaluator = new ClusteringEvaluator();
			double silhouette = evaluator.evaluate(s.transform(testData));
			
			System.out.println("silhouette score: " + silhouette);
		});
		
		inputVectorsForProcessing.unpersist();

	}
	
	private static List<String> listDirectory(String dir) {
		File dirFile = new File(dir);
		return Arrays.asList(dirFile.listFiles()).parallelStream()
				.filter(f -> f.getName().split("\\.")[0].endsWith("B"))
				.map(f -> f.getPath()).collect(Collectors.toList());
	}

	protected static List<String> getResourceFiles(String path) throws IOException {
		List<String> filenames = new ArrayList<>();
		System.out.println(ClassPathUtils.toFullyQualifiedName(FraudDetectionMain.class, SAMPLE_FILE));

		try (InputStream in = getResourceAsStream(path);
				BufferedReader br = new BufferedReader(new InputStreamReader(in))) {
			String resource;

			while ((resource = br.readLine()) != null) {
				System.out.println("Added: " + resource);
				filenames.add(resource);
			}
		}

		return filenames;
	}

	private static InputStream getResourceAsStream(String resource) {
	    final InputStream in
	            = getContextClassLoader().getResourceAsStream(resource);

	    return in == null ? getContextClassLoader().getResourceAsStream(resource) : in;
	}

	private static ClassLoader getContextClassLoader() {
	    return Thread.currentThread().getContextClassLoader();
	}

	protected static List<String> locateSampleFile() throws IOException {
		Enumeration<URL> fileUrl = ClassLoader.getSystemResources(SAMPLE_FILE);
		if (fileUrl == null) {
			throw new FileNotFoundException("The following file could not be located on the classpath: " + SAMPLE_FILE);
		}

		List<String> urlsToReturn = new ArrayList<>();
		for (URL url = null; fileUrl.hasMoreElements(); url = fileUrl.nextElement()) {
			urlsToReturn.add(url.getFile());
		}
			
		return urlsToReturn;
	}

	private static Dataset<Row> parseFile(List<String> sampleFile) throws MalformedURLException {
		List<String> filesToLoad = new ArrayList<String>(sampleFile.size());
		for(String aFile : sampleFile) {
			String fileLocation = new File(aFile).toURI().toURL().toString();
			System.out.println(fileLocation);
			filesToLoad.add(fileLocation);
		}
		SparkSession session = SparkSupport.sparkSession;
		
		Dataset<Row> dataset = session.read().format("csv")
				.option("header", "true").schema(SparkSupport.schema).load(filesToLoad.toArray(new String[0]));
		return dataset;
	}
}
