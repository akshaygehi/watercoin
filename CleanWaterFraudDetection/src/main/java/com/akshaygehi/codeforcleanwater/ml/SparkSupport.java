package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkSupport {

	static final SparkConf conf = new SparkConf().setMaster("local").setAppName("CleanWater");
	
	static final JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(conf));

}
