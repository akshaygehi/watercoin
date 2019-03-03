package com.akshaygehi.codeforcleanwater.ml;

import java.io.File;
import java.io.FileNotFoundException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class FraudDetectionMain {

	private static final String SAMPLE_FILE = "LrgInd/IND100B.csv";

	static final List<FraudDetectionStrategy> strategies = Arrays.asList(new KmeansFraudDetectionStrategy(),
			new GaussianMixtureDetectionStrategy());

	public static void main(String[] args) throws Exception {
		VectorParser parser = new VectorParser();

		String file = locateSampleFile();

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
		
		strategies.forEach(s -> s.trainModel(inputVectorsForProcessing));

		inputVectorsForProcessing.unpersist();

	}

	private static String locateSampleFile() throws FileNotFoundException {
		URL fileUrl = ClassLoader.getSystemResource(SAMPLE_FILE);
		if (fileUrl == null) {
			throw new FileNotFoundException("The following file could not be located on the classpath: " + SAMPLE_FILE);
		}

		String file = fileUrl.getFile();
		return file;
	}

	private static Dataset<Row> parseFile(String sampleFile) throws MalformedURLException {
		String fileLocation = new File(sampleFile).toURI().toURL().toString();
		SparkSession session = SparkSupport.sparkSession;
		
		StructType schema = new StructType(new StructField[] {
			new StructField("id", DataTypes.StringType, true, Metadata.empty()),
			new StructField("time", DataTypes.LongType, true, Metadata.empty()),
			new StructField("vol", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("ph", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("solids", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("hardness", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("oil", DataTypes.DoubleType, true, Metadata.empty()),
			new StructField("bod", DataTypes.DoubleType, true, Metadata.empty())
		});
		
		Dataset<Row> dataset = session.read().format("csv")
				.option("header", "true").schema(schema).load(fileLocation);
		return dataset;
	}
}
