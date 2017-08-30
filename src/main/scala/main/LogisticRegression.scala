import java.io.{File, FileOutputStream, PrintWriter}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{LabeledPoint, OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}


Logger.getLogger("org").setLevel(Level.WARN)
Logger.getLogger("akka").setLevel(Level.WARN)
val logger = Logger.getLogger(getClass.getName)


val sample3 = 100.0
val sample2 = 500.0
val sample1 = 250.0
val samples = Seq(50.0, 250.0, 500.0)
//anzahl untersuchungen
val aU = 6
val lAU = aU - 1

val sampleList = List(sample1, sample2, sample3)

//some values & variables
val dir = "path"
val similarietiesFile1 = dir + "training_musicbrainz1_2.csv"
val similarietiesFile2 = dir + "training_musicbrainz1_3.csv"
val similarietiesFile3 = dir + "training_musicbrainz1_4.csv"
val similarietiesFile4 = dir + "training_musicbrainz1_5.csv"

val csvOptions = new java.util.HashMap[String, String]()
csvOptions.put("header", "true")
csvOptions.put("sep", ";")
csvOptions.put("inferSchema", "true")
val spark = SparkSession.builder().appName("ExampleSession").config("spark.master", "local[*]").getOrCreate()

val similarities1 = spark.read.options(csvOptions).csv(similarietiesFile1)
val similarities2 = spark.read.options(csvOptions).csv(similarietiesFile2)
val similarities3 = spark.read.options(csvOptions).csv(similarietiesFile3)
val similarities4 = spark.read.options(csvOptions).csv(similarietiesFile4)

val similarities = similarities1.union(similarities2).union(similarities3).union(similarities4).toDF()


val booleanToDouble: (Boolean => Double) = (arg: Boolean) => {if (arg) 1.0 else 0.0}
val sqlfunc = udf(booleanToDouble)

def createLabeledPoint(label: Double, t :Double, a : Double, alb: Double) : LabeledPoint = {
  val vec = new DenseVector(Array(t,a,alb))
  val l = new LabeledPoint(label, vec)
  return l
}


println("Das Schema ist folgendes: ")
similarities.printSchema()


//println("Die erste Zeile ist: ")
val colnames = similarities.columns
val firstrow = similarities.head(1)(0)
println("Beispieldatensatz:")
for (ind <- Range(1, colnames.length)) {
  print(colnames(ind) + ": '")
  println(firstrow(ind) + "' ")
}

import spark.implicits._
val similaritiesData = similarities.withColumn("verified2", sqlfunc(col("verified")))
val dataAll = similaritiesData.select(similaritiesData("verified2").as("label")
  , $"title-title"
  , $"artist-artist"
  , $"album-album"
)
//for all sample data sets
val similaritiesData1 = similarities1.withColumn("verified2", sqlfunc(col("verified")))
val dataAll1 = similaritiesData1.select(similaritiesData1("verified2").as("label")
  , $"title-title"
  , $"artist-artist"
  , $"album-album"
)
val similaritiesData2 = similarities2.withColumn("verified2", sqlfunc(col("verified")))
val dataAll2 = similaritiesData2.select(similaritiesData2("verified2").as("label")
  , $"title-title"
  , $"artist-artist"
  , $"album-album"
)
val similaritiesData3 = similarities3.withColumn("verified2", sqlfunc(col("verified")))
val dataAll3 = similaritiesData3.select(similaritiesData3("verified2").as("label")
  , $"title-title"
  , $"artist-artist"
  , $"album-album"
)
val similaritiesData4 = similarities4.withColumn("verified2", sqlfunc(col("verified")))
val dataAll4 = similaritiesData4.select(similaritiesData4("verified2").as("label")
  , $"title-title"
  , $"artist-artist"
  , $"album-album"
)


//drop all data which we won't use any more
val data = dataAll.na.drop()
val data1 = dataAll1.na.drop()
val data2 = dataAll2.na.drop()
val data3 = dataAll3.na.drop()
val data4 = dataAll4.na.drop()

val countAll = data.count()

//divide all data
val trues = data.filter("verified = 1.0")
val falses = data.filter("verified = 0.0")

val truesAmount = trues.count()
val falsesAmount = falses.count()




val assembler = new VectorAssembler().setInputCols(Array("title-title", "artist-artist", "album-album")).setOutputCol("features")
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.0)

val pipeline = new Pipeline().setStages(Array(
  assembler
  , lr
))
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.3, 0.1, 0.01)).build()

val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)



def getColumnFromTraining(number: Int):String = number match {
  case 0 => return "all;all"
  case 1 => return "1_2;1_2"
  case 2 => return "1_2;1_3"
  case 3 => return "1_2;1_4"
  case 4 => return "1_2;1_5"
  case 5 => return "1_2;all"
  case _ => return "not supported"
}

val p = "path_to_file"
val pw = new PrintWriter(new FileOutputStream(new File(p),true))
try pw.write("sample;trainingdata;testdata;Precision;Recall;Accuracy;FMeasure\n")


// for loop for different sample sets
for(sample <- samples){
  val shareTrue : Double = sample / truesAmount
  val shareTrue2 : Double = 1.0 - shareTrue

  val shareFalse : Double = sample / falsesAmount
  val shareFalse2 : Double = 1.0 - shareFalse

  val Array(trainingTrue, testTrue) = trues.randomSplit(Array(shareTrue, shareTrue2), seed = 12345)
  val Array(trainingFalse, testFalse) = falses.randomSplit(Array(shareFalse, shareFalse2), seed = 12345)

  val trainingData = trainingTrue.union(trainingFalse)
  val testData = testTrue.union(testFalse)


  //for model of first dataset
  val trues1 = data1.filter("verified = 1.0")
  val falses1 = data1.filter("verified = 0.0")

  val truesAmount1 = trues1.count()
  val falsesAmount1 = falses1.count()

  val shareTrue5 : Double = sample / truesAmount1
  val shareTrue6 : Double = 1.0 - shareTrue5

  val shareFalse5 : Double = sample / falsesAmount1
  val shareFalse6 : Double = 1.0 - shareFalse5

  val Array(trainingTrue1, testTrue1) = trues1.randomSplit(Array(shareTrue5, shareTrue6), seed = 12345)
  val Array(trainingFalse1, testFalse1) = falses1.randomSplit(Array(shareFalse5, shareFalse6), seed = 12345)

  val trainingData1 = trainingTrue1.union(trainingFalse1)
  val testData1 = testTrue1.union(testFalse1)



  // Run cross-validation, and choose the best set of parameters.
  val cvModel = cv.fit(trainingData)



  /** use one model for all training data */
  // --- all --- //
  val model = pipeline.fit(trainingData)
  val results = model.transform(testData)
  /** use a model for each separated testingset */
  // --- 1 --- //
  val model1 = pipeline.fit(trainingData1)
  val results1 = model1.transform(testData1)
  // --- 2 --- //
  val results2 = model1.transform(data2)
  // --- 3 --- //
  val results3 = model1.transform(data3)
  // --- 4 --- //
  val results4 = model1.transform(data4)
  /** test the first model against all data */
  // --- 1-all --- //
  val results1all = model1.transform(data)


  // Model Evaluation
  val predictionAndLabels = results.select("prediction", "label").as[(Double, Double)].rdd
  val metrics = new MulticlassMetrics(predictionAndLabels)

  /*
  // Model Evaluation
  val predictionAndLabels1 = results1.select("prediction", "label").as[(Double, Double)].rdd
  val metrics1 = new MulticlassMetrics(predictionAndLabels1)
  // Model Evaluation
  val predictionAndLabels2 = results2.select("prediction", "label").as[(Double, Double)].rdd
  val metrics2 = new MulticlassMetrics(predictionAndLabels2)
  // Model Evaluation
  val predictionAndLabels3 = results3.select("prediction", "label").as[(Double, Double)].rdd
  val metrics3 = new MulticlassMetrics(predictionAndLabels3)
  // Model Evaluation
  val predictionAndLabels4 = results4.select("prediction", "label").as[(Double, Double)].rdd
  val metrics4 = new MulticlassMetrics(predictionAndLabels4)
  // Model Evaluation
  val predictionAndLabels1all = results1all.select("prediction", "label").as[(Double, Double)].rdd
  val metrics1all = new MulticlassMetrics(predictionAndLabels1all)
  */
  val tps = new Array[Double](aU)
  val tns = new Array[Double](aU)
  val fps = new Array[Double](aU)
  val fns = new Array[Double](aU)

  val precision = new Array[Double](aU)
  val recall = new Array[Double](aU)
  val accuracy = new Array[Double](aU)
  val fmeasure = new Array[Double](aU)


  //all
  tps(0) = results.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(0) = results.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(0) = results.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(0) = results.filter("prediction = 0.0").filter("label = 1.0").count()
  //1
  tps(1) = results1.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(1) = results1.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(1) = results1.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(1) = results1.filter("prediction = 0.0").filter("label = 1.0").count()
  //2
  tps(2) = results2.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(2) = results2.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(2) = results2.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(2) = results2.filter("prediction = 0.0").filter("label = 1.0").count()
  //3
  tps(3) = results3.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(3) = results3.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(3) = results3.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(3) = results3.filter("prediction = 0.0").filter("label = 1.0").count()
  //4
  tps(4) = results4.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(4) = results4.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(4) = results4.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(4) = results4.filter("prediction = 0.0").filter("label = 1.0").count()
  //4
  tps(5) = results1all.filter("prediction = 1.0").filter("label = 1.0").count()
  tns(5) = results1all.filter("prediction = 0.0").filter("label = 0.0").count()
  fps(5) = results1all.filter("prediction = 1.0").filter("label = 0.0").count()
  fns(5) = results1all.filter("prediction = 0.0").filter("label = 1.0").count()

  for(i <- 0 to 5){
    println("TP: "+tps(i))
    println("TN: "+tns(i))
    println("FP: "+fps(i))
    println("FN: "+fns(i))
    precision(i) = tps(i) / (tps(i) + fps(i))
    recall(i) = tps(i) / (tps(i) + fns(i))
    accuracy(i) = (tps(i) + tns(i)) / (tns(i) + tps(i) + fns(i) + fps(i))
    fmeasure(i) = (2 * precision(i) * recall(i)) / (precision(i) + recall(i))
  }
  /*
  val precision = tp / (tp + fp)
  val recall = tp / (tp + fn)
  val accuracy = (tp + tn) / (tp + tn + fp + fn)
  */

  println("Bei der Berechnung kamen folgende Werte raus:")
  for(a <- 0 to 5){
    pw.write("" +
      2*sample +
      ";"+
      getColumnFromTraining(a) +
      ";"+precision(a)+
      ";"+recall(a)+
      ";"+accuracy(a)+
      ";"+fmeasure(a)+
      "\n")
    println(a)
    println("Precision: "+precision(a))
    println("Recall: "+ recall(a))
    println("Accuracy: " + accuracy(a))
    println("FMeasure: " + fmeasure(a))
  }

  println("Confusion matrix:")
  try {
    println(metrics.confusionMatrix.toString())
  } catch {
    case e: StringIndexOutOfBoundsException => logger.error("StringIndexOutOfBoundsException ")
    case unknown: Throwable => logger.error("Got this unknown exception: " + unknown)


  }

  val metricsBinary = new BinaryClassificationMetrics(predictionAndLabels)
  val roc = metricsBinary.roc().collect
}


pw.close()
//accuracy = (tp + tn) / (all)
//


//p=path, s=string,
def writeToFile(p: String, s: String, num: Double): Unit = {
  val pw = new PrintWriter(new FileOutputStream(new File(p),true))
  try pw.write(s + ", " + num + "\n") finally pw.close()
}