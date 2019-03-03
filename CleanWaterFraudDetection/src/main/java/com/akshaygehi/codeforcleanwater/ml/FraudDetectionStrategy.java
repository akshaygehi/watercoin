package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public interface FraudDetectionStrategy {
	
	void trainModel(Dataset<Row> data);
	
	Dataset<Row> transform(Dataset<Row> data);
	
	PipelineStage getStage();
	
	void saveModel();
	
	void loadModel();

}
