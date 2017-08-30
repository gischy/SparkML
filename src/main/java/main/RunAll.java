package main;

import java.io.IOException;

/**
 * Created by Kirchgeorg on 27.07.2017.
 */
public class RunAll {

	public static void main(String[] args) throws IOException {
		LogisticRegressionTrainAndTest1_2Sample100_500_1000.main(null);
		LogisticRegressionTrain1_2AndTest1_3_Sample100_500_1000.main(null);
		LogisticRegressionTrain1_2AndTest1_4_Sample100_500_1000.main(null);
		LogisticRegressionTrain1_2AndTest1_5_Sample100_500_1000.main(null);
	}
	
}
