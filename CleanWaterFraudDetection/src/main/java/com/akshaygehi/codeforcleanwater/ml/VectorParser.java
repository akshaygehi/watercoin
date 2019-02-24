package com.akshaygehi.codeforcleanwater.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class VectorParser {
	
	public JavaRDD<Vector> parseData(JavaRDD<String> data) {
		JavaRDD<List<String>> data2 = data.filter(l -> !l.startsWith("id")).map((String l) -> {
			
			List<String> cols = new ArrayList<>(Arrays.asList(l.split(",")));
			String idCol = cols.get(0); // IND100A
			// Convert sensor ID to an integer 
			int sensorId = idCol.charAt(idCol.length() - 1);
			// TODO : Replace prefix using config
			cols.set(0, idCol.substring(0, idCol.length() - 2).replace("IND", "0"));
			cols.add(1, String.valueOf(sensorId));
			
			return cols;
		}).filter(cols -> Double.parseDouble(cols.get(3)) > 0);
		
		return data2.map(cols -> {
			
			int size = cols.size();
			double[] values = new double[size - 2];
			  for (int i = 2; i < size; i++) {
			    values[i - 2] = Double.parseDouble(cols.get(i));
			    
			    if(i == 2) {
			    	// Reduce the precision of the timestamp to minutes
			    	// so that ML can match better
			    	values[i - 2] = values[i - 2] / (1000 * 60);
			    }
			    
			  }
			  
			  return Vectors.dense(values);
		});
	}
	

}
