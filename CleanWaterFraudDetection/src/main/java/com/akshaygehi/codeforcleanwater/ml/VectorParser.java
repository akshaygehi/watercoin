package com.akshaygehi.codeforcleanwater.ml;

import static org.apache.spark.sql.functions.substring;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class VectorParser {
	
	public Dataset<Row> parseData(Dataset<Row> input) {
		input = input.withColumn("sensorType", substring(input.col("id"), 6, 7).as("sensorType"));
		input = input.withColumn("idNew", substring(input.col("id"), 3, 6).as("idNew"));
		
		return input;

//			
//			int size = cols.size();
//			double[] values = new double[size - 2];
//			  for (int i = 2; i < size; i++) {
//			    values[i - 2] = Double.parseDouble(cols.get(i));
//			    
//			    if(i == 2) {
//			    	// Reduce the precision of the timestamp to minutes
//			    	// so that ML can match better
//			    	values[i - 2] = values[i - 2] / (1000 * 60);
//			    }
//			    
//			  }
//			  
//			  return Vectors.dense(values);
//		});
	}
	

}
