# SparkML

This program utilizes Apache Spark and their machine learning library (MLlib) to train and test a binomial logistic regression model with sample data from musicbrainz. The model is trained with a given number of true and false links (100,500,1000) and the given similarity vectors between two music records. The model is tested with given similarity vectors and an unknown link attribute to predict if two records are classified as similar and are predicted as linked or not. Different metrics e.g. true/false positive rate, confusion matrix, accuracy, precision, ... are calculated to evaluate the logistic regression model and the classification results.

The implementation using the spark java api can be found inside "src/java" and an implementation using the spark scala api is located inside "src/scala".

## Java Implementation

### Requirements

- Maven is required to build the project
- For executing Spark on Windows the file winutils.exe is required
- Paths need to be configured inside Config.java

### Usage

- Using the command "mvn install" will build the project and create an executable JAR file
- The JAR file can be executed after the build process from the main directory using the command "java -jar target/sparkml-1.0.jar" (running the command from inside the target folder will throw an exception)
- The application can also be started by running the class RunAll.java from the IDE
- Training data is located inside the "training_data" directory
- Metrics for each iteration with a different sample size will be stored inside the "metriken" directory
- Models for each iteration with a different sample size will be stored inside the "models" directory

### Results

A summary pdf file "Metriken Logistische Regression.pdf" containing all metrics and the corresponding graphs for the precision recall curve and the roc curve can be found inside the "metriken" directory.

## Scala Implementation

The scala script can be loaded into the spark-shell using ":load <path-to-file>". The script should previously adapted by changing the paths to the files and the output folder. The script executes for each element defined in the sample collection(sampleList) the following task:
- dividing true links and false links into a trainingset and a testset using a randomSplit function
- train a model using the given file
- test the model against the given datasets
- evaluate the results by getting true positives, true negatives, false positives, false negatives and calculate the metrics precision, recall, accuracy and f1 measure for each comparison
- saving the calculated metrics with the name of the trainingset and testset and also the amount of entries used for training the model