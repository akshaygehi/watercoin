package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;

public interface FraudDetectionStrategy {
	
	void trainModel(JavaRDD<Vector> data);
	
	void saveModel();
	
	void loadModel();

}
