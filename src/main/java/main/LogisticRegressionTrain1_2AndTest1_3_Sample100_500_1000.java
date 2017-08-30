package main;/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class LogisticRegressionTrain1_2AndTest1_3_Sample100_500_1000 {

	public static List<LabeledPoint> pointListTraining;
	public static List<LabeledPoint> pointListTest;

	public static int[] sampleSizes = {100,500,1000};

	public static StringBuilder sb;

	public static void main(String[] args) throws FileNotFoundException {

		for (int sampleSize : sampleSizes) {

			sb = new StringBuilder();

			pointListTraining = new ArrayList<>();
			pointListTest = new ArrayList<>();

			// spark configuration
			System.setProperty("hadoop.home.dir", Config.hadoopPath);
			SparkConf conf = new SparkConf().setAppName("LogisticRegressionExample").setMaster("local[*]");
			JavaSparkContext jsc = new JavaSparkContext(conf);
			jsc.setLogLevel("ERROR");
			SQLContext jsql = new SQLContext(jsc);

			// load training data
			Dataset<Row> trainingDataset = jsql.read().format("csv").option("inferSchema", "true").option("header", "true").option("sep", ";").load(Config.pathTrainingData1);

			// filtering links
			Dataset<Row> verifiedLinks = trainingDataset.filter("verified = true");
			Dataset<Row> unVerifiedLinks = trainingDataset.filter("verified = false");

			Row[] linksTrue = (Row[]) verifiedLinks.collect();
			System.out.println("datensatz gesamtanzahl links true: " + linksTrue.length);


			Row[] linksFalse = (Row[]) unVerifiedLinks.collect();
			System.out.println("datensatz gesamtanzahl false links: " + linksFalse.length);

			int numberOfLinks = sampleSize/2;

			List<LabeledPoint> labeledPointsTrueTrain = new ArrayList<LabeledPoint>();

			for (int i = 0; i < linksTrue.length && i < numberOfLinks; i++) {
				labeledPointsTrueTrain.add(createLabeledPoint(linksTrue[i]));
			}

			List<LabeledPoint> labeledPointsFalseTrain = new ArrayList<LabeledPoint>();

			for (int i = 0; i < linksFalse.length && i < numberOfLinks; i++) {
				labeledPointsFalseTrain.add(createLabeledPoint(linksFalse[i]));
			}

			pointListTraining.addAll(labeledPointsTrueTrain);
			pointListTraining.addAll(labeledPointsFalseTrain);

			log("anzahl links true (training): " + labeledPointsTrueTrain.size());
			log("anzahl links false (training): " + labeledPointsFalseTrain.size());

			log("--> gesamtanzahl der eingelesen trainingsdaten: " + pointListTraining.size());
			Dataset<Row> trainingData = jsql.createDataFrame(jsc.parallelize(pointListTraining), LabeledPoint.class);

			// load test data
			Dataset<Row> testDataset = jsql.read().format("csv").option("inferSchema", "true").option("header", "true").option("sep", ";").load(Config.pathTrainingData2);

			testDataset.foreach((ForeachFunction<Row>) row -> {
				pointListTest.add(createLabeledPoint(row));
			});

			log("--> gesamtanzahl der eingelesen testdaten: " + pointListTest.size());

			Dataset<Row> testData = jsql.createDataFrame(jsc.parallelize(pointListTest), LabeledPoint.class);

			// logistic regression instance (estimator)
			LogisticRegression lr = new LogisticRegression();

			// logistic regression parameters
			lr.setMaxIter(10)
			  .setRegParam(0);

			System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");

			// learn logistic regression model with training data
			LogisticRegressionModel model = lr.fit(trainingData);

			log("model coefficients: " + model.coefficients());
			log("model threshold: " + model.getThreshold());

			double coefficientTitel = model.coefficients().apply(0);
			double coefficientArtist = model.coefficients().apply(1);
			double coefficientAlbum = model.coefficients().apply(2);

			log("coefficient titel: " + coefficientTitel);
			log("coefficient artist: " + coefficientArtist);
			log("coefficient album: " + coefficientAlbum);

			// apply trained model to test data
			Dataset<Row> results = model.transform(testData);

			Dataset<Row> testRows = results.select("prediction", "label");

			MulticlassMetrics multiclassMetrics = new MulticlassMetrics(testRows);
			log("confusion matrix:");
			log(multiclassMetrics.confusionMatrix().toString());

			log("true positive: " +multiclassMetrics.confusionMatrix().apply(1,1));
			log("true negative: " +multiclassMetrics.confusionMatrix().apply(0,0));
			log("false positive: " +multiclassMetrics.confusionMatrix().apply(1,0));
			log("false negative: " +multiclassMetrics.confusionMatrix().apply(0,1));

			log("accuracy: " + multiclassMetrics.accuracy());
			log("precision: " + multiclassMetrics.precision());

			log("f measure: " + multiclassMetrics.fMeasure());
			log("weighted f measure: " + multiclassMetrics.weightedFMeasure());

			log("weighted precision: " + multiclassMetrics.weightedPrecision());
			log("recall: " + multiclassMetrics.recall());
			log("weighted recall: " + multiclassMetrics.weightedRecall());
			log("weighted true positive rate: " + multiclassMetrics.weightedTruePositiveRate());
			log("weighted false positive rate: " + multiclassMetrics.weightedFalsePositiveRate());

			BinaryClassificationMetrics binaryClassificationMetrics = new BinaryClassificationMetrics(testRows);
			log("precision-recall-curve: " +binaryClassificationMetrics.pr().toJavaRDD().collect());
			log("roc-curve: " +binaryClassificationMetrics.roc().toJavaRDD().collect());

			BinaryLogisticRegressionSummary summary = new BinaryLogisticRegressionSummary(results, "probability", "label", "features");

			jsc.stop();

			// write metrics
			writeFile(LogisticRegressionTrain1_2AndTest1_3_Sample100_500_1000.class.getName()+"_iteration"+sampleSize, sb.toString());
		}
	}

	/**
	 * Creates a @{@link LabeledPoint} object from a row of our datasets.
	 * @param row row containing the linked attribute and 3 similarity vectors
	 * @return @{@link LabeledPoint}
	 */
	private static LabeledPoint createLabeledPoint(Row row) {
		boolean labelBoolean = row.getBoolean(2);
		double label = labelBoolean == true ? 1.0 : 0.0;
		Vector features = new DenseVector(new double[] {row.getDouble(3), row.getDouble(4), row.getDouble(5)});
		return new LabeledPoint(label, features);
	}

	/**
	 * Outputs and logs necessary information.
	 * @param log line which should be appended to the log
	 */
	private static void log(String log) {
		System.out.println(log);
		sb.append(log);
		sb.append("\n");
	}

	/**
	 * Writes a file with a given filename and payload to the output directory.
	 * @param fileName file name
	 * @param payload content of the file
	 * @throws FileNotFoundException
	 */
	private static void writeFile(String fileName, String payload) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File(Config.outputPath + fileName));
		pw.write(payload);
		pw.close();
	}

}