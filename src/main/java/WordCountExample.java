

import org.apache.hadoop.hdfs.server.namenode.dfsclusterhealth_jsp;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Vector;

/**
 * Created by fkirchge on 30.04.2017.
 */
public class WordCountExample {

    public static void main(String[] args) {

        String logFile = "YOUR_SPARK_HOME/README.md"; // Should be some file on your system

        SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("Simple Application");

        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> logData = sc.textFile(logFile).cache();

        JavaRDD<String> textFile = sc.textFile("file://C:/Users/fkirchge/Desktop/testrepos/sparkml/eingabefile");

        JavaPairRDD<String, Integer> counts = textFile
                .flatMap(s -> Arrays.asList(s.split(" ")).iterator())
                .mapToPair(word -> new Tuple2<>(word, 1))
                .reduceByKey((a, b) -> a + b);

        counts.coalesce(1).saveAsTextFile("file:///C:/Users/fkirchge/Desktop/testrepos/sparkml/ausgabefile");

    }


}
