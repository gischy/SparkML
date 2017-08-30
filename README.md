# SparkML

This program utilizes Apache Spark and their machine learning library (MLlib) to train and test a binomial logistic regression model with sample data from musicbrainz. The model is trained with a given number of true and false links (100,500,1000) and the given similarity vectors between two music records. The model is tested with given similarity vectors and an unknown link attribute to predict if two records are classified as similar and are predicted as linked or not. Different metrics e.g. true/false positive rate, confusion matrix, accuracy, precision, ... are calculated to evaluate the logistic regression model and the classification results.

The implementation using the spark java api can be found inside "src/java" and an implementation using the spark scala api is located inside "src/scala".

## Requirements

- Maven is required to build the project
- For executing Spark on Windows the file winutils.exe is required
- Paths need to be configured inside Config.java

## Usage

- Using the command "mvn install" will build the project and create an executable JAR file
- The JAR file can be executed using the command "java -jar target/sparkml-1.0.jar"
- The application can also be started by running the class RunAll.java from the IDE
- Training data is located inside the "training_data" directory
- Metrics for each iteration with a different sample size will be stored inside "metriken" directory

## Results

A summary pdf file "Metriken Logistische Regression.pdf" containing all metrics and the corresponding graphs for the precision recall curve and the roc curve can be found inside the "metriken" directory.
