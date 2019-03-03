package com.akshaygehi.codeforcleanwater.ml;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class SparkSupport {

	static final SparkConf conf = new SparkConf().setMaster("local").setAppName("CleanWater");
	
	static final SparkSession sparkSession = SparkSession.builder().config(conf).getOrCreate();
	
	static final JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkContext.getOrCreate(conf));
	
	static final StructType schema = new StructType(
			new StructField[] { new StructField("id", DataTypes.StringType, true, Metadata.empty()),
					new StructField("time", DataTypes.LongType, true, Metadata.empty()),
					new StructField("vol", DataTypes.DoubleType, true, Metadata.empty()),
					new StructField("ph", DataTypes.DoubleType, true, Metadata.empty()),
					new StructField("solids", DataTypes.DoubleType, true, Metadata.empty()),
					new StructField("hardness", DataTypes.DoubleType, true, Metadata.empty()),
					new StructField("oil", DataTypes.DoubleType, true, Metadata.empty()),
					new StructField("bod", DataTypes.DoubleType, true, Metadata.empty()) });
	
}
