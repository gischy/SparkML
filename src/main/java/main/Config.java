package main;

/**
 * Created by Kirchgeorg on 30.08.2017.
 */
public class Config {

	public static String hadoopPath = "C:/hadoop/";

	public static String pathTrainingData1 = System.getProperty("user.dir")+"/training_data/training_musicbrainz1_2.csv";
	public static String pathTrainingData2 = System.getProperty("user.dir")+"/training_data/training_musicbrainz1_3.csv";
	public static String pathTrainingData3 = System.getProperty("user.dir")+"/training_data/training_musicbrainz1_4.csv";
	public static String pathTrainingData4 = System.getProperty("user.dir")+"/training_data/training_musicbrainz1_5.csv";

	public static String outputPathMetrics = System.getProperty("user.dir") + "/metriken/";
	public static String outputPathModels = System.getProperty("user.dir") + "/models/";



}
