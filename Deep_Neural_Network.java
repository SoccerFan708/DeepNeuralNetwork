import java.io.Serializable;

public class Deep_Neural_Network implements Serializable{
   private static final long serialVersionUID = 1L;


   int networkDepth; //Network Depth
   int inputNodes;//Input Nodes
   int hiddenNodes;//Hidden Nodes
   int outputNodes;//Output Nodes
   int[] hiddenLayersNodes;
   int[] Nodes; //Nodes per layer
   double[][][] W; //Weights
   double[][][] b; //Biases
   double[][][] a; //Unactivated layer inputs
   double[][][] h; //Activated layer outputs
   double[][][] layerErrors; //Layer Errors
   double[][] x; //Input
   double[][] y; //Target
   String networkType = "classification";
   String finalLayerActivationType = "softmax"; //Activation function type
   String errorFunction = "cross entropy";
   int costFunction = 2; //Cost function type
   double LEARNING_RATE = .01;
   double initialLearningRate;
   double[][][] TRAINING_DATASET;
   double[][][] TRAINING_DATASET_TARGETS;
   private int[] TRAINING_DATASET_CLASSES;
   double[][][] VALIDATION_DATASET;
   double[][][] VALIDATION_DATASET_TARGETS;
   private int[] VALIDATION_DATASET_CLASSES;
   double[][][] TEST_DATASET;
   double[][][] TEST_DATASET_TARGETS;
   private int[] TEST_DATASET_CLASSES;
   int dataClasses;
   int[][] classPrediction;
   double[] recall;
   double[] precision;
   double[] F1_score;
   double[][] regression_test_results = new double[1][1];
   double regression_avg_difference = 1;
   boolean regularize = true;
   double regularization_factor = 0.000009;
   
   




   public Deep_Neural_Network(){ 
   }
   
   public Deep_Neural_Network(int depth, int inNodes, int hidNodes, int outNodes, double learningRate){
      networkDepth = depth;
      inputNodes = inNodes;
      hiddenNodes = hidNodes;
      outputNodes = outNodes;
      
      LEARNING_RATE = learningRate;
      initialLearningRate = learningRate;
      
      dataClasses = outputNodes;
      classPrediction = new int[dataClasses][dataClasses];
      recall = new double[dataClasses];
      precision = new double[dataClasses];
      F1_score = new double[dataClasses];
     
      Nodes = new int[networkDepth];
      Nodes[0] = inputNodes;
      for(int n=1;n<Nodes.length;n++){
         Nodes[n] = hiddenNodes;
      }
      Nodes[Nodes.length-1] = outputNodes;
      INITIATE_WEIGHTS();
      INITIATE_BIASES();
      a = new double[networkDepth][Nodes[0]][1];
      for(int n=0;n<Nodes.length;n++){
         a[n] = new double[Nodes[n]][1];
      }
      
      
      h = new double[networkDepth][Nodes[0]][1];
      for(int n=0;n<Nodes.length;n++){
         h[n] = new double[Nodes[n]][1];
      }
   }
   
   
   
   public Deep_Neural_Network(int[] layerNodes, double learningRate){
      networkDepth = layerNodes.length;
      inputNodes = layerNodes[0];
      
      int[] hidNodes = new int[layerNodes.length-2];
      for(int i=1;i<layerNodes.length-1;i++){
         hidNodes[i-1] = layerNodes[i];
      }
      
      hiddenLayersNodes = hidNodes;
      hiddenNodes = hidNodes[0];
      outputNodes = layerNodes[layerNodes.length-1];
      
      LEARNING_RATE = learningRate;
      initialLearningRate = learningRate;
      
      dataClasses = outputNodes;
      classPrediction = new int[dataClasses][dataClasses];
      recall = new double[dataClasses];
      precision = new double[dataClasses];
      F1_score = new double[dataClasses];
     
      Nodes = new int[networkDepth];
      Nodes[0] = inputNodes;
      for(int n=0;n<hiddenLayersNodes.length;n++){
         Nodes[n+1] = hiddenLayersNodes[n];
      }
      Nodes[Nodes.length-1] = outputNodes;
      INITIATE_WEIGHTS();
      INITIATE_BIASES();
      a = new double[networkDepth][Nodes[0]][1];
      for(int n=0;n<Nodes.length;n++){
         a[n] = new double[Nodes[n]][1];
      }
      
      
      h = new double[networkDepth][Nodes[0]][1];
      for(int n=0;n<Nodes.length;n++){
         h[n] = new double[Nodes[n]][1];
      }
   }
   
   public void set_network_type(String netType, String actType, String errorFunc){
      networkType = netType;
      switch(networkType){
         case "regression":
            regularize = false;
            break;
      }
      
      finalLayerActivationType = actType;
      errorFunction = errorFunc;
   }
   
   
   double[][][] gW; //Weight Gradients
   double[][][] gB; //Bias Gradients
   //Array[Row][Column]
   
   private void INITIATE_WEIGHTS(){
   
      W = new double[networkDepth][Nodes[0]][Nodes[0]];
      gW = new double[networkDepth][Nodes[0]][Nodes[0]];
     
      
      //System.out.println("W[0]:"+W[0].length+"_"+W[0][0].length);
         
      for(int i=1;i<W.length;i++){
         W[i] = new double[Nodes[i-1]][Nodes[i]];
         gW[i] = new double[Nodes[i-1]][Nodes[i]];
         //System.out.println("W["+i+"]:"+W[i][0].length+"_"+W[i].length);
      }
      
   
      //W[W.length-1] = new double[Nodes[Nodes.length-1]][Nodes[Nodes.length-2]];
      //gW[W.length-1] = new double[Nodes[Nodes.length-1]][Nodes[Nodes.length-2]];
      
      for(int i=0;i<W.length;i++){
         W[i] = initial_random_fill(W[i], 2);
         String label = "WEIGHTS FROM LAYER "+i;
         reveal(W[i], label, 3);
      }
      
   }
   
   private void INITIATE_BIASES(){
      b = new double[networkDepth][Nodes[0]][1];
      gB = new double[networkDepth][Nodes[0]][1];
      layerErrors = new double[networkDepth][Nodes[0]][1];
      for(int i=0;i<b.length;i++){
         b[i] = new double[Nodes[i]][1];
         gB[i] = new double[Nodes[i]][1];
         layerErrors[i] = new double[Nodes[i]][1];
      }
      //b[b.length-1] = new double[Nodes[Nodes.length-1]][1];
      //Initiate biases with 0.1;
      for(int i=0;i<b.length;i++){
         for(int j=0;j<b[i].length;j++){
            b[i][j][0] = 0.1;
         }
      }
   }
   
   private void INITIATE_UNACTIVATED_VALUES(){
   }
   
   private void INITIATE_ACTIVATED_VALUES(){
   }
   
   double[][] GRADIENT;
   
   public void provideTrainingData(double[][][] trainingData, double[][][] trainingTargets){
      TRAINING_DATASET = trainingData;
      TRAINING_DATASET_TARGETS = trainingTargets;
   }
   
   public void provideValidationData(double[][][] validationData, double[][][] validationTargets){
      VALIDATION_DATASET = validationData;
      VALIDATION_DATASET_TARGETS = validationTargets;
   }
   public void provideTestData(double[][][] testData, double[][][] testTargets){
      TEST_DATASET = testData;
      TEST_DATASET_TARGETS = testTargets;
      
      //TEST_DATASET = TRAINING_DATASET;
      //TEST_DATASET_TARGETS = TRAINING_DATASET_TARGETS;
   }
   
   public void provideData(double[][][] passedData, double[][][] passedDataTargets){
   //SHUFFLE
      //int[] sequence = shuffle(passedData.length);
      //passedData = shuffle(passedData, sequence);
      //passedDataTargets = shuffle(passedDataTargets, sequence);
      //
      int trainPartition = getPartitionIndex(passedData.length, .50);
      System.out.println("TRAINING DATA");
      provideTrainingData(transposeDataForNetwork(separateData(passedData, 0, trainPartition)), transposeDataForNetwork(separateData(passedDataTargets, 0, trainPartition)));
      TRAINING_DATASET_CLASSES = analyseClassDistribution(TRAINING_DATASET_TARGETS);
      int validationPartition = getPartitionIndex(passedData.length, .75);
      System.out.println("VALIDATION DATA");
      provideValidationData(transposeDataForNetwork(separateData(passedData, trainPartition, validationPartition)), transposeDataForNetwork(separateData(passedDataTargets, trainPartition, validationPartition)));
      VALIDATION_DATASET_CLASSES = analyseClassDistribution(VALIDATION_DATASET_TARGETS);
      System.out.println("TEST DATA");
      provideTestData(transposeDataForNetwork(separateData(passedData, trainPartition, passedData.length)), transposeDataForNetwork(separateData(passedDataTargets, trainPartition, passedDataTargets.length)));
      TEST_DATASET_CLASSES = analyseClassDistribution(TEST_DATASET_TARGETS);
   }
   
   public void provideData(double[][][] passedData, double[][][] passedDataTargets, boolean shuff){
   //SHUFFLE
      if(shuff){
         int[] sequence = shuffle(passedData.length);
        // passedData = shuffle(passedData, sequence);
         //passedDataTargets = shuffle(passedDataTargets, sequence);
      //
      }
      int trainPartition = getPartitionIndex(passedData.length, .50);
      System.out.println("TRAINING DATA");
      provideTrainingData(transposeDataForNetwork(separateData(passedData, 0, trainPartition)), transposeDataForNetwork(separateData(passedDataTargets, 0, trainPartition)));
      TRAINING_DATASET_CLASSES = analyseClassDistribution(TRAINING_DATASET_TARGETS);
      int validationPartition = getPartitionIndex(passedData.length, .75);
      System.out.println("VALIDATION DATA");
      provideValidationData(transposeDataForNetwork(separateData(passedData, trainPartition, validationPartition)), transposeDataForNetwork(separateData(passedDataTargets, trainPartition, validationPartition)));
      VALIDATION_DATASET_CLASSES = analyseClassDistribution(VALIDATION_DATASET_TARGETS);
      System.out.println("TEST DATA");
      provideTestData(transposeDataForNetwork(separateData(passedData, trainPartition, passedData.length)), transposeDataForNetwork(separateData(passedDataTargets, trainPartition, passedDataTargets.length)));
      TEST_DATASET_CLASSES = analyseClassDistribution(TEST_DATASET_TARGETS);
   }
   
   public double[][] getClassDistribution(){
      double[][] classDistributions = new double[3][dataClasses];
      for(int i=0;i<classDistributions.length;i++){
      
      }
      
      return classDistributions;
   }
   
   private int[] analyseClassDistribution(double[][][] dataTargets){
      int[] classes = new int[dataClasses];
      
      for(int i=0;i<dataTargets.length;i++){
         for(int c=0;c<classes.length;c++){
            if(dataTargets[i][c][0] == 1){
               classes[c] += 1;
            }
         }
      }
     
      for(int i=0;i<classes.length;i++){
         System.out.println(classes[i]+", ");
      }
      
      return classes;
   }
   
   private void calculateF1Score(){
   
      for(int p=0;p<dataClasses;p++){
         for(int i=0;i<classPrediction.length;i++){
            //System.out.println("recall add "+classPrediction[i][p]);
            recall[p] += classPrediction[i][p];
            //System.out.println("precision add "+classPrediction[p][i]);
            precision[p] += classPrediction[p][i];
         }
      }
      
      for(int p=0;p<classPrediction.length;p++){
         if(precision[p]<=0){
            precision[p] = 1;
         }
         if(recall[p]<=0){
            recall[p] = 1;
         }
         
         recall[p] = classPrediction[p][p]/recall[p];
         precision[p] = classPrediction[p][p]/precision[p];
         double f1Sum = recall[p]+precision[p];
         if(f1Sum > 0){
            F1_score[p] = 2 * ((recall[p]*precision[p])/(f1Sum));
         }
         
         
         System.out.println("recall["+p+"] = "+recall[p]);
         System.out.println("precision["+p+"] = "+precision[p]);
         System.out.println("F1_score["+p+"] = "+F1_score[p]);
               
      }
      
      int ctr = 0;
      //System.out.println("test classes = "+TEST_DATASET_CLASSES.length);
      //System.out.println("test data = "+TEST_DATASET.length);
      for(int d=0;d<TEST_DATASET_CLASSES.length;d++){
         //System.out.println("test["+d+"] = "+TEST_DATASET_CLASSES[d]);
         double chk = ((double)(TEST_DATASET_CLASSES[d])/((double)TEST_DATASET.length));
         //System.out.println("check = "+chk);
         if(chk >= 0.1){
            ctr += 1;
         }
      }
      networkF1Score = 0;
      for(int f=0;f<F1_score.length;f++){
         networkF1Score += F1_score[f];
      }
      networkF1Score = networkF1Score/ctr;
      //System.out.println("F1_score = "+networkF1Score);
   
   }
   
   
   
   
   private  double[][][] transposeDataForNetwork(double[][][] passedData){
      for(int t=0;t<passedData.length;t++){
         passedData[t] = transpose(passedData[t]);
      }
   
      return passedData;
   }
   
   private int getPartitionIndex(int dataLength, double divisionRate){
      int partition;
      partition = (int)(dataLength*divisionRate);
      //System.out.println("Total data: "+dataLength+" Separation Point: "+partition);
      return partition;
   }
   
   private double[][][] separateData(double[][][] passedData, int begin, int end){
      double[][][] dataSlice = new double[end-begin][1][passedData[0][0].length];
      for(int i=begin;i<end;i++){
         dataSlice[i-begin] = passedData[i];
      }
      return dataSlice;
   }
  
   double[][] Epoch_results;
   double networkAccuracy = 0; 
   int epochs;
   int batchSize;
   double accuracyThreshold;
   boolean accuracyReached = false;
   
   double networkF1Score;
 
   boolean normalize_data = true;     
   
   public void setTrainingParameters(int epochs, int batchSize, double accuracyThreshold){
      Epoch_results = new double[epochs][4];
      this.epochs = epochs;
      this.batchSize = batchSize;
      this.accuracyThreshold = accuracyThreshold;
   }
   
   public double[][] trainEpoch(int currentEpoch){
      double[][][] trainingData = TRAINING_DATASET;
      double[][][] targets = TRAINING_DATASET_TARGETS;
      int[] sequence;
      int start = 0;
      int end = 0;
      double[][] bold_G;
      double validationAccuracy = 0;  
      int batches = (trainingData.length/batchSize);
      int stoppedEpoch = Epoch_results.length;
      double[][] tempEpochResults = Epoch_results;
      int e = currentEpoch;
      
      
      sequence = shuffle(trainingData.length);
      double currentEpoch_tErrors = 0;
      for(int b=0;b<batches;b++){
         start = b*batchSize;
         if((start+batchSize)<trainingData.length){
            end = start+batchSize;
         }else{
            end = trainingData.length;
         }
         double[][][] sampleGradients = new double[(end-start)][2][2];
         double[][][] dataLoss = new double[(end-start)][2][2];
         double tErrors = 0;
         for(int i=start;i<end;i++){
            //System.out.println("Set "+(i-start)+" of batch "+(b+1));
            double[][] sampledData = trainingData[sequence[i]-1];
            double[][] sampledTarget = targets[sequence[i]-1];
            Stat st = new Stat();
            st.analyze(sampledData);
            double max = st.max[0][0];
            double min = st.min[0][0];
            double mean = st.mean[0][0];
            double sd = st.standard_deviation[0][0];
            if(normalize_data){
               sampledData = normalizeByRow(sampledData, max, min, mean, sd);
               sampledTarget = normalizeByRow(sampledTarget, max, min, mean, sd);
            }
            
            
            double[][] modelOutput = forwardPropagate(sampledData);
            reveal(modelOutput, "TRAINING OUTPUT", 3);
            reveal(sampledTarget, "TRAINING TARGET", 3);
            dataLoss[i-start]  = getNodeError(errorFunction, modelOutput, sampledTarget);
            reveal(dataLoss[i-start], "LOSS", 3);
            sampleGradients[i-start] = derive_cost_function_WRT_pre_activation_function_output(errorFunction, finalLayerActivationType, modelOutput, sampledTarget);
            reveal(sampleGradients[i-start], "LOSS DERIVATIVE", 3);
            tErrors += getTotalError(errorFunction, dataLoss[i-start]);
         }
         bold_G = getAvg(sampleGradients);
         //annealLearningRate(e);   
         backProp(bold_G);
         currentEpoch_tErrors += tErrors/batchSize;
      }
      
      
         
      double[][][] validationErrors = new double[VALIDATION_DATASET.length][2][2];
      double vError = 0;
      for(int v=0;v<VALIDATION_DATASET.length;v++){
      
         double[][] sampledData = VALIDATION_DATASET[v];
         double[][] sampledTarget = VALIDATION_DATASET_TARGETS[v];
         Stat st = new Stat();
         st.analyze(sampledData);
         double max = st.max[0][0];
         double min = st.min[0][0];
         double mean = st.mean[0][0];
         double sd = st.standard_deviation[0][0];
         if(normalize_data){
            
            reveal(sampledData, "SAMPLE BEFORE", 4);
            reveal(sampledTarget, "TARGET BEFORE", 4);
         
            sampledData = normalizeByRow(sampledData, max, min, mean, sd);
            sampledTarget = normalizeByRow(sampledTarget, max, min, mean, sd);
            reveal(sampledData, "SAMPLE AFTER", 4);
            reveal(sampledTarget, "TARGET AFTER", 4);
         }
      
         double[][] modelOutput = forwardPropagate(sampledData);
         reveal(modelOutput, "VALIDATION OUTPUT", 3);
         reveal(sampledTarget, "VALIDATION TARGET", 3);
         validationErrors[v] = getNodeError(errorFunction, modelOutput, sampledTarget);
         vError += getTotalError(errorFunction, validationErrors[v]);
      }
      vError = vError/VALIDATION_DATASET.length;
      
      
      switch(networkType){
         case "classification":
            classificationTest();
            if((networkAccuracy >= accuracyThreshold) && !accuracyReached){
               stoppedEpoch = e+1;
               accuracyReached = true;
            }
            if(NaNEncountered){
               stoppedEpoch = e;
            }
         
            if(e < tempEpochResults.length){
               tempEpochResults[e][0] = currentEpoch_tErrors/batches;
               tempEpochResults[e][1] = vError;
               tempEpochResults[e][2] = networkAccuracy;
               tempEpochResults[e][3] = networkF1Score;
               //System.out.println("\n"+"Epoch = "+e);
               //System.out.println("Training Loss = "+tempEpochResults[e][0]);
               //System.out.println("Validation Loss = "+tempEpochResults[e][1]);
            }
            //System.out.println("Stopped at "+stoppedEpoch);
            //System.out.println("Temp length: "+tempEpochResults.length);
            Epoch_results = new double[stoppedEpoch][4]; 
            for(int s=0;s<stoppedEpoch;s++){
               Epoch_results[s] = tempEpochResults[s];
            }
            break;
            
         case "regression":
            double[][][] testErrors = new double[TEST_DATASET.length][2][2];
            regression_test_results = new double[TEST_DATASET.length][3];
            double[] testDifference = new double[TEST_DATASET.length];
            double tError = 0;
            for(int t=0;t<TEST_DATASET.length;t++){
            
               double[][] sampledData = TEST_DATASET[t];
               double[][] sampledTarget = TEST_DATASET_TARGETS[t];
               Stat st = new Stat();
               st.analyze(sampledData);
               double max = st.max[0][0];
               double min = st.min[0][0];
               double mean = st.mean[0][0];
               double sd = st.standard_deviation[0][0];
               if(normalize_data){
                  sampledData = normalizeByRow(sampledData, max, min, mean, sd);
                  sampledTarget = normalizeByRow(sampledTarget, max, min, mean, sd);
               }
            
               double[][] modelOutput = forwardPropagate(sampledData);
               reveal(sampledData, "TEST INPUT", 1);
               reveal(modelOutput, "TEST OUTPUT", 1);
               reveal(sampledTarget, "TEST TARGET", 1);
               testErrors[t] = getNodeError(errorFunction, modelOutput, sampledTarget);
               tError += getTotalError(errorFunction, testErrors[t]);
               testDifference[t] = regressionDifference(modelOutput, sampledTarget);
               regression_test_results[t][0] = regressionVectorAvg(sampledTarget);
               regression_test_results[t][1] = regressionVectorAvg(modelOutput);
               regression_test_results[t][2] = testDifference[t];
               //System.out.println("difference: "+ testDifference[t]);
               
               if(normalize_data){
                  regression_test_results[t][0] = regressionVectorAvg(denormalizeByRow(sampledTarget, max, min, mean, sd));
                  regression_test_results[t][1] = regressionVectorAvg(denormalizeByRow(modelOutput, max, min, mean, sd));
                  regression_test_results[t][2] = regressionDifference(denormalizeByRow(modelOutput, max, min, mean, sd), denormalizeByRow(sampledTarget, max, min, mean, sd));
                  //System.out.println("difference: "+ regressionDifference(denormalizeByRow(modelOutput, max, min, mean, sd), denormalizeByRow(sampledTarget, max, min, mean, sd)));
               }
            }
            
            //double testDiff = 0;
            /*
            for(int f=0;f<testDifference.length;f++){
               regression_avg_difference += testDifference[f];
            }
            regression_avg_difference = regression_avg_difference/TEST_DATASET.length;
            */
            for(int f=0;f<regression_test_results.length;f++){
               regression_avg_difference += regression_test_results[f][2];
            }
            /*Dividing by 100 to set 100 as a max for graphing purposes*/
            regression_avg_difference = (regression_avg_difference/TEST_DATASET.length)/100;
            
            
            tError = tError/TEST_DATASET.length;
           
            if(e < tempEpochResults.length){
               tempEpochResults[e][0] = currentEpoch_tErrors/batches;
               tempEpochResults[e][1] = vError;
               tempEpochResults[e][2] = tError;
               tempEpochResults[e][3] = regression_avg_difference;
               //System.out.println("\n"+"Epoch = "+e);
               //System.out.println("Training Loss = "+tempEpochResults[e][0]);
               //System.out.println("Validation Loss = "+tempEpochResults[e][1]);
            }
            //System.out.println("Stopped at "+stoppedEpoch);
            //System.out.println("Temp length: "+tempEpochResults.length);
            Epoch_results = new double[stoppedEpoch][4]; 
            for(int s=0;s<stoppedEpoch;s++){
               Epoch_results[s] = tempEpochResults[s];
            }
         
            break;
      }
              
      return Epoch_results;
            
   
   }
   
   public void annealLearningRate(int t){
      LEARNING_RATE = initialLearningRate/(1+(t/(50)));
   
   }
   
   
   
   public double[][] normalizeByRow(double[][] passedData, double max, double min, double mean, double sd){
      double[][] normalizedData = new double[passedData.length][passedData[0].length];
      for(int i=0;i<passedData.length;i++){
         for(int j=0;j<passedData[i].length;j++){
            normalizedData[i][j] = normalize(passedData[i][j], max, min, mean, sd);
         }
      }
      return normalizedData;
   }
   
   public double[][] denormalizeByRow(double[][] passedData, double max, double min, double mean, double sd){
      double[][] denormalizedData = new double[passedData.length][passedData[0].length];
      for(int i=0;i<passedData.length;i++){
         for(int j=0;j<passedData[i].length;j++){
            denormalizedData[i][j] = denormalize(passedData[i][j], max, min, mean, sd);
         }
      }
      return denormalizedData;
   }
   
   public double normalize(double passedValue, double max, double min, double mean, double sd){
      double normalizedValue = 0;
      //normalizedValue = (passedValue - min)/(max - min);
      normalizedValue = ((passedValue - min)/(max - min));
      //normalizedValue = ((passedValue - mean)/(sd));
      return normalizedValue;
   }
   
   public double denormalize(double normalizedValue, double max, double min, double mean, double sd){
      double denormalizedValue = 0;
      denormalizedValue = normalizedValue*(max - min)+min;
      //denormalizedValue = normalizedValue*(sd)+mean;
      return denormalizedValue;
   }
   
   
   public void classificationTest(){
      int correctPredictions = 0;
      classPrediction = new int[dataClasses][dataClasses];
      for(int t=0;t<TEST_DATASET.length;t++){
         double[][] modelOutput = forwardPropagate(TEST_DATASET[t]);
            //System.out.println("ModelOutput "+modelOutput[0].length);
         reveal(TEST_DATASET[t], "TEST INPUT", 1);
         modelOutput = getPrediction(modelOutput);
         reveal(modelOutput, "TEST PREDICTION", 1);
         reveal(TEST_DATASET_TARGETS[t], "TEST TARGET", 1);
               //System.out.print("Prediction");
            //printMatrix(transpose(modelOutput));
            //System.out.print("Test Target");
            //printMatrix(transpose(TEST_DATASET_TARGETS[t]));
            
         int predictedClass = getClass(modelOutput);
         int targetClass = getClass(TEST_DATASET_TARGETS[t]);
            
            //if(compareVectors(modelOutput, TEST_DATASET_TARGETS[t])){
         if(predictedClass == targetClass){
            correctPredictions += 1;
               //System.out.println("Correct Prediction");
         }  
         classPrediction[predictedClass][targetClass] += 1;  
      }
         
      calculateF1Score();
         
      double total = TEST_DATASET.length;
      networkAccuracy = correctPredictions/total;
            //System.out.println("Test Accuracy = "+networkAccuracy); 
            //System.out.println(correctPredictions+"/"+total);  
            //printMatrix(classPrediction);
   
   
            
     
   }
   
   private double regressionDifference(double[][] output, double[][] target){
      double difference = 0;
      double op = 0;
      double tg = 0;
      for(int i=0;i<output.length;i++){
         for(int j=0;j<output[i].length;j++){
            difference += Math.abs(target[i][j] - output[i][j]);
         }
      }
      
      difference = difference/target.length;
   
      return difference;
   }
   
   private double regressionVectorAvg(double[][] vector){
      double avg = 0;
   
      for(int i=0;i<vector.length;i++){
         for(int j=0;j<vector[i].length;j++){
            avg += vector[i][j];
         }
      }
   
      avg = avg/vector.length;
   
      return avg;
   }
   
   boolean compareVectors(double[][] vectorOne, double[][] vectorTwo){
      boolean answer = false;
   
      if(vectorOne.length == vectorTwo.length && vectorOne[0].length == vectorTwo[0].length){
         int counter = 0;
         for(int i=0;i<vectorOne.length;i++){
            for(int j=0;j<vectorOne[0].length;j++){
                 
               if(vectorOne[i][j] == vectorTwo[i][j]){
                  //System.out.println("vectorOne: "+vectorOne[0][0]+","+vectorOne[1][0]+","+vectorOne[2][0]);
                  //System.out.println("vectorTwo: "+vectorTwo[0][0]+","+vectorTwo[1][0]+","+vectorTwo[2][0]);
                  counter += 1;
                  //System.out.println("correct");
               }
            }
         }
         
         if(counter == vectorOne.length){
            answer = true;
            //System.out.println("all correct");
         }
      }
   
   
      return answer;
   }
   
   /*****************************NOTE************************************
   for each epoch
         for each training data instance
            propagate error through network
            adjust weights
            calculate the accuracy over the training data
         for each validation data instance
            calculate accuracy over validation data
            if validation accuracy threshold reached
               exit training
            else
               continue training   
   
   ***********************************************************************/
   
   
   
   double[][] getPrediction(double[][] output){
      double[][] prediction = new double[output.length][output[0].length];
      double[][] temp = transpose(output);
      double maximum = findMax(temp[0]);
      for(int i=0;i<output.length;i++){
         //System.out.println("Output: "+output[0][0]+","+output[1][0]+","+output[2][0]+". max: "+maximum);
         //System.out.println(output[i].length);
         for(int j=0;j<output[0].length;j++){
         
            if(output[i][j] ==  maximum){
               prediction[i][j] = 1;
            }else{
               prediction[i][j] = 0;
            }
         }
      }
      
      return prediction;
   }
   
   public int getClass(double[][] vector){
      int cl = -1;
   
      for(int v=0;v<vector.length;v++){
         for(int e=0;e<vector[v].length;e++){
            //System.out.println("vector["+v+"]["+e+"]:"+vector[v][e]);
            if(vector[v][e] == 1){
               cl = v;
            }
         }
      }
   
      return cl;
   }
   
   double getTotalError(String type, double[][] loss){
      double cost = 0;
   
      switch(type){
         case "mean squared error":
         //MSE
            for(int i=0;i<loss.length;i++){
               cost +=  Math.pow((loss[i][0]), 2);
            }
            cost = cost/(2*(loss.length));
            
            break;
         case "cross entropy":
         //CROSS ENTROPY
            for(int i=0;i<loss.length;i++){
               cost += loss[i][0];
            }
            cost = cost;
            //System.out.println("COST ==> "+cost);
            break;
         case "log-likelihood":
         //LOG-LIKELIHOOD
         
            break;
      }
   
      return cost;
   }
   
   public double[][] getNodeError(String type, double[][] output, double[][] target){
      double[][] loss = new double[output.length][output[0].length];
      
      switch(type){
         case "mean squared error":
         //MEAN SQUARED ERROR
            for(int i=0;i<output.length;i++){
               loss[i][0] =  (output[i][0] - target[i][0]);
            }
            break;
         case "cross entropy":
            //CROSS ENTROPY
            double oup = 0;
            for(int i=0;i<output.length;i++){
               loss[i][0] = -((target[i][0]*Math.log(output[i][0])))+((1-target[i][0])*Math.log(1-output[i][0]));
            }
            break;
         case "log-likelihood":
         //LOG-LIKELIHOOD COST
            for(int i=0;i<output.length;i++){
               loss[i][0] =  (output[i][0] - target[i][0]);
            }
            break;
      }
      
      return loss;
   }
   
   
   
   public double[][] getAvg(double[][][] batch){
      double[][] avg = new double[batch[0].length][batch[0][0].length];
      for(int l=0;l<batch.length;l++){
         for(int j=0;j<avg.length;j++){
            avg[j][0] += batch[l][j][0];
         }
      }
      
      for(int j=0;j<avg.length;j++){
         avg[j][0] = avg[j][0]/batch.length;
      }
      
      return avg;
   }
   

   
   //*********************************************************************************
   public int[] shuffle(int size){
      int[] sequence = new int[size];
   
      for(int i=0;i<sequence.length;i++){
         int randomNumber = getRandomNumber(size);
         boolean found = check(randomNumber, sequence);
         while(found){
            randomNumber = getRandomNumber(size);
            found = check(randomNumber, sequence);
         }
         sequence[i] = randomNumber;
      }
      
      return sequence;
   }
   
   public double[][][] shuffle(double[][][] passedData, int[] sequence){
      double[][][] shuffledData = passedData;
      //int[] sequence = shuffle(passedData.length);
   /* 
      for(int i=0;i<sequence.length;i++){
        // shuffledData[i] = passedData[sequence[i]-1];
      }
   */
      return shuffledData;
   }
   
   public boolean check(int number, int[] sequence){
      boolean found = false;
      for(int s=0;s<sequence.length;s++){
         if(sequence[s] == number){
            found = true;
         }
      }
   
      return found;
   } 
   
   public int getRandomNumber(int roof){
      int number = 0;
      number = (int)(Math.random()*roof)+1;
      return number;
   }
//**************************************************************************************************
   public double[][] query(double[][] inp){
      double[][] output = new double[1][1];
      double[][] input = inp;
      
      Stat st = new Stat();
      st.analyze(input);
      double max = st.max[0][0];
      double min = st.min[0][0];
      double mean = st.mean[0][0];
      double sd = st.standard_deviation[0][0];
      if(normalize_data){
         input = normalizeByRow(input, max, min, mean, sd);
      }
      output = forwardPropagate(input); 
      if(normalize_data){
         output = denormalizeByRow(output, max, min, mean, sd);
      }
      //System.out.println("query "+output.length);  
      return output;
   }
   
   
   
   

   
   public void backProp(double[][] loss){
      double[][] errors = loss;
      compute_layer_errors(errors, networkDepth-1);
      compute_bias_and_weight_gradients();
      update_parameters();
      
      
   }
   
   
   public double[][] clip_gradient(double[][] passedG, double[] threshold){
      double [][] boldG = passedG;
   
      double l2_norm = 0;
   
      for(int i=0;i<passedG.length;i++){
         l2_norm += Math.pow(passedG[i][0], 2);
      }
      System.out.println("l2 sum = "+l2_norm);
      l2_norm = Math.pow(l2_norm, 0.5);
      System.out.println("l2 sqrt = "+l2_norm);
      if(l2_norm <= threshold[0]){
         System.out.println(l2_norm+" <= "+threshold[0]);
         reveal(passedG, "Gradient Pre Clip", 3);
         for(int i=0;i<passedG.length;i++){
            boldG[i][0] = (threshold[0]/l2_norm)*passedG[i][0];
         }
         reveal(boldG, "Gradient Post Clip", 3);
      }
   
      
      if(l2_norm >= threshold[1]){
         System.out.println(l2_norm+" >= "+threshold[1]);
         reveal(passedG, "Gradient Pre Clip", 3);
         for(int i=0;i<passedG.length;i++){
            boldG[i][0] = (threshold[1]/l2_norm)*passedG[i][0];
         }
         reveal(boldG, "Gradient Post Clip", 3);
      }
   
      return boldG;
   }
 
   
       
   
   public double getRegularizer(double regularizingFactor, double[][] nodeWeights){
      double regularizer = 0;
   
      for(int j=0;j<nodeWeights.length;j++){
         for(int k=0;k<nodeWeights[j].length;k++){
            regularizer += Math.pow(nodeWeights[j][k], 2);
         }
      }
      regularizer = regularizer*regularizingFactor;
   
      return regularizer;
   }
   
         
   boolean show = false;
   int revealLevel = 1;
   
   private void reveal(double[][] Matrix, String label, int level){
      if(show && ((level <= revealLevel)||(revealLevel == 10))){
         System.out.print(label);
         printMatrix(transpose(Matrix));
      }
   }
   
   
   
   public double[][] derive_cost_function_WRT_pre_activation_function_output(String cost_type, String activation_type,  double[][] output, double[][] target){
      double[][] derivativeVector = new double[output.length][output[0].length];
      switch(cost_type){
         case "cross entropy": 
            switch(activation_type){
               case "softmax":
                  for(int i=0;i<output.length;i++){
                     //derivativeVector[i][0] = ((target[i][0]*(1/output[i][0]))+(1-target[i][0])*(1/(1-output[i][0])));
                     derivativeVector[i][0] = target[i][0] - output[i][0];
                  }
                  break;
            }
            break;
            
         case "mean squared error":
            switch(activation_type){
            
               case "sigmoid":
                  for(int i=0;i<output.length;i++){   
                     double c1 = output[i][0]*(1 - output[i][0]);
                     //System.out.println("c1: "+c1);
                     derivativeVector[i][0] = -(output[i][0]-target[i][0])*c1;//-(target[i][0] - output[i][0]);
                     reveal(derivativeVector, "DERIVATIVE LOSS", 2);
                  }
                  break;
               case "ReLU":
                  
                  for(int i=0;i<output.length;i++){   
                     double c1 = (output[i][0]-target[i][0]);
                     //if(a[networkDepth-1][i][0]>0 || (a[networkDepth-1][i][0]<0 && c1>0)){
                     derivativeVector[i][0] = -c1*(output[i][0]);
                     //}else{
                        //derivativeVector[i][0] = 0;
                     //reveal(derivativeVector, "DERIVATIVE LOSS", 2);
                     //}
                  }
                  break;
               case "linear":
                  for(int i=0;i<output.length;i++){   
                     double c1 = (output[i][0]-target[i][0]);
                     //if(a[networkDepth-1][i][0]>0 || (a[networkDepth-1][i][0]<0 && c1>0)){
                     derivativeVector[i][0] = -c1;
                     //}else{
                        //derivativeVector[i][0] = 0;
                     //reveal(derivativeVector, "DERIVATIVE LOSS", 2);
                     //}
                  }
               
                  break;
            }
         
            break;
      }
      return derivativeVector;
   }
   
   
   private void compute_layer_errors(double[][] final_layer_errors, int final_layer){
      layerErrors[final_layer] = final_layer_errors;
      for(int f=final_layer-1;f>0;f--){
         reveal(layerErrors[f+1], "before LAYER["+(f+1)+"] ERROR", 3);
         reveal(W[f+1], "before LAYER["+(f+1)+"] WEIGHTS", 3);
         layerErrors[f] = dot(W[f+1], layerErrors[f+1]);
         reveal(layerErrors[f], "after LAYER["+f+"] ERROR", 3);
         //if(h[f].length>0){
         layerErrors[f] = multiply(layerErrors[f], max_derivative(layerErrors[f], h[f]));
         //}
         reveal(layerErrors[f], "after multiply LAYER["+f+"] ERROR", 3);  
      }
      for(int i=0;i<layerErrors.length;i++){
         reveal(layerErrors[i], "ERROR FOR LAYER "+i, 3);
      }
   }
   
   private void compute_bias_and_weight_gradients(){
   //BIAS
      for(int f=1;f<layerErrors.length;f++){
         gB[f] = layerErrors[f];
      }
   //WEIGHTS
      for(int f=1;f<layerErrors.length;f++){
         reveal(gW[f], "Weight gradients["+f+"]", 3);
         reveal(h[f-1], "h["+(f-1)+"]", 2);
         reveal(layerErrors[f], "LayerErrors["+f+"]", 2);
         //gW[f] = multiply(h[f-1], layerErrors[f]);
         for(int i=0;i<(h[f-1].length);i++){
            for(int j=0;j<layerErrors[f].length;j++){
               gW[f][i][j] = h[f-1][i][0]*layerErrors[f][j][0];
            }
         }
         reveal(gW[f], "Weight gradients["+f+"]", 3);
      }
   }
   
   private void update_parameters(){
      for(int i=(networkDepth-1);i>0;i--){
         if(regularize){
            double regularizer = getRegularizer(regularization_factor, W[i]);
            gW[i] = add_variable(regularizer,  gW[i]);
         }
         reveal(gW[i], "GRADIENTS AFTER REGULARIZING", 3);
         //Modify weights with gradient
         reveal(W[i], "WEIGHTS BEFORE ADDING WEIGHT GRADIENTS", 3);
         W[i] = add(W[i], scalar(LEARNING_RATE, gW[i]));
         reveal(W[i], "WEIGHTS AFTER ADDING WEIGHT GRADIENTS", 3);
         
         reveal(b[i], "BIASES BEFORE ADDING BIAS GRADIENTS", 3);
         reveal(gB[i], "BIAS GRADIENTS", 3);
         b[i] = add(b[i], scalar(LEARNING_RATE, gB[i]));
         reveal(b[i], "BIASES AFTER ADDING BIAS GRADIENTS", 3);
      }
   }
  
   
   private double[][] max_derivative(double[][] gradients, double[][] layerInput){
      double[][] derivative = new double[gradients.length][gradients[0].length];
      for(int i=0;i<gradients.length;i++){
         for(int j=0;j<gradients[0].length;j++){
            if(layerInput[i][j]>=0 || (layerInput[i][j]<0 && gradients[i][j]>0)){
               derivative[i][j] = 1;
            }else{
               derivative[i][j] = 0;
            }
         }
      }
   
      return derivative;
   }
   
   
   double[][] preActivationOutput;
   private double[][] forwardPropagate(double[][] inp){
      //System.out.println("h0: "+h[0].length);
      x = inp;
      h[0] = x;
      //double[][] temp = new double[1][1];
      for(int k=1;k<networkDepth;k++){
        // System.out.println("h"+(k-1)+": "+h[k-1].length+"_"+h[k-1][0].length);
         //System.out.println("W"+k+": "+W[k].length+"_"+W[k][0].length);
         //reveal(h[k-1], "h["+(k-1)+"] BEFORE MULTIPLYING WITH W["+k+"]", 2);
         //reveal(W[k], "W["+k+"]", 2);
         //reveal(b[k], "b["+k+"]", 2);
         checkForNaN(h[k-1], "h[k-1]");
         checkForNaN(W[k], "W[k]");
         a[k] = dot(transpose(W[k]), h[k-1]);
         //checkForNaN(a[k], "a[k]");
         //reveal(a[k], "a["+k+"] BEFORE ADDING BIASES", 2);
         //System.out.println("temp at"+k+": "+temp.length+"_"+temp[0].length);
         //System.out.println("b["+(k)+"]: "+b[k].length+"_"+b[k][0].length);
         a[k] = add(b[k], a[k]);
         //reveal(a[k], "a["+k+"] AFTER ADDING BIASES", 2);
         //System.out.println("a["+k+"]: "+a[k].length+"_"+a[k][0].length);
         if(k == (networkDepth-1)){
            preActivationOutput = a[k];
            reveal(preActivationOutput, "OUTPUT BEFORE FINAL ACTIVATION", 3);
            h[k] = activation_function(finalLayerActivationType, a[k]);
            //double[][] tempH = transpose(h[k]);
            
         }else{
            h[k] = activation_function("ReLU", a[k]);
         }
         //reveal(h[k], "LAYER OUTPUT AFTER ACTIVATING FUNCTION", 1);
      }
      reveal(h[networkDepth-1], "FINAL OUTPUT", 3);
      return h[networkDepth-1];
   
   
   }
   boolean NaNEncountered = false;
   private void checkForNaN(double[][] a, String name){
      for(int o=0;o<a.length;o++){
         Double temp = Double.valueOf(a[o][0]);
         if(temp.isNaN()){
            System.out.println("NaN encountered at "+name);
            printMatrix(a);
            NaNEncountered = true;
            break;
         }
      }
   
   }
   
         //Populate weight matrix with random weights.
   private double[][] initial_random_fill(double[][] lw, int base){
      double[][]temp = lw;
      for(int i=0;i<temp.length;i++){
         for(int j=0;j<temp[0].length;j++){
            //double w = (Math.random()*(.3));
            double w = (Math.random()-.5)*((1/Math.sqrt((lw.length))));
            //System.out.println("random w "+w);
            if(Double.valueOf(w).isNaN()){
               //System.out.println("NaN weight");
            }
         
            temp[i][j] =    w;            
         }
      }
      return  temp;
   }
   
   //Sigmoid function is used as the Activation Function
   private double[][] activation_function(String type, double[][] inp){
      double[][] oup = new double[inp.length][1];
      
      
      switch(type){
         case "sigmoid":
         //SIGMOID
            for(int i=0;i<inp.length;i++)
               for(int j=0;j<inp[0].length;j++)
                  oup[i][j] = sigmoid(inp[i][j]);
                 
            break;
         case "ReLU":
         //ReLU
                      /*
               Notes: ReLUs
               z=activation;
               W=weights;
               x=input;
               
               z = max{0, (W*x)}
               for all z values that are greater than 0, the gradient for the weights is going to be the derivative
               of the weights with respect to value presented for activation:   
               
               dW = (z > 0, x)
               
               */ 
            for(int i=0;i<inp.length;i++)
               for(int j=0;j<inp[0].length;j++)
                  oup[i][j] = max(inp[i][j]);
            break;
            
         case "softmax":
         //SOFTMAX
            oup = softmax(inp);
            break;
         case "linear":
         //LINEAR
            for(int i=0;i<inp.length;i++)
               for(int j=0;j<inp[0].length;j++)
                  oup[i][j] = inp[i][j];
            break;
      }
      
      return oup;
   }
   
   private double sigmoid(double x){
      double y = 0;
      double negativeX = -(x);
      y = 1/(1+Math.pow(Math.E, negativeX));
      return y;
   }
   
   private double max(double x){
      double m = 0;
      if(x>0){
         m = x;
      }
      return m;
   }
   
   double[][] softmax(double[][] scores){
      double[][] softMaxScores = new double[scores.length][scores[0].length];
      double sum = 0;
      double max = findMax(scores);
     //System.out.println("MAX Score: "+max);
      for(int i=0;i<scores.length;i++){
         
         softMaxScores[i][0] = Math.exp(scores[i][0]-max);
         sum += softMaxScores[i][0];
            
      }
      
      //System.out.println("SUM: "+sum);
      for(int i=0;i<scores.length;i++){
         
         softMaxScores[i][0] = softMaxScores[i][0]/sum;
         
      }
      return softMaxScores;
   }
   
   double findMax(double[][] data){
      double max = 0;
   
      double[] temp = new double[data.length];
   
      for(int d=0;d<data.length;d++){
         temp[d] = data[d][0];
      }
      max = findMax(temp);
      Double Max = Double.valueOf(max);
      if(Max.isInfinite()){
         System.out.println("Infinite:");
         //printMatrix(data);
         max = 0;
      }
   
      return max;
   
   }
   
   
   // rows(M) X column(N)..Matrix Dot multiplication
   private double[][] dot(double[][] A, double[][] B){
   
      double[][] C = new double[0][0];
      if(A[0].length == B.length){
         C = new double[A.length][B[0].length];
         
         for (int i = 0; i < C.length; i++)
            for (int j = 0; j < C[0].length; j++)
               for (int k = 0; k < A[0].length; k++)
                  C[i][j] += (A[i][k] * B[k][j]);
      }
   
      return C;
   }
   
   private double[][] multiply(double[][] A, double[][] B){
   
      double[][] C = new double[0][0];
      if(A.length == B.length){
         C = new double[A.length][B[0].length];
         
         for (int i = 0; i < C.length; i++)
            for (int j = 0; j < C[0].length; j++)
               C[i][j] += (A[i][j] * B[i][j]);
      }
   
      return C;
   }
   
   //Scalar multiplication
   private double[][] scalar(double sc, double[][] A){
     
      for (int i = 0; i < A.length; i++)
         for (int j = 0; j < A[0].length; j++)
            A[i][j] = A[i][j] * sc;
      
      return A;
   }
   
   private double[][] subtract(double[][] A, double[][] B){
      double[][] C = new double[1][1];
   
      if(A.length == B.length && A[0].length == B[0].length){
         C = new double[A.length][A[0].length];
         for(int i=0;i<A.length;i++)
            for(int j=0;j<A[0].length;j++)
               C[i][j] = A[i][j] - B[i][j];
      }
      
      return C;
   }
   
   private double[][] subtract_from_variable(double v, double[][] A){
      double[][] C = new double[1][1];
   
      C = new double[A.length][A[0].length];
      for(int i=0;i<A.length;i++)
         for(int j=0;j<A[0].length;j++)
            C[i][j] = v - A[i][j];
      
      
      return C;
   }
   
   private double[][] add_variable(double v, double[][] A){
      double[][] C = new double[1][1];
   
      C = new double[A.length][A[0].length];
      for(int i=0;i<A.length;i++)
         for(int j=0;j<A[0].length;j++)
            C[i][j] = A[i][j] + v;
      
      
      return C;
   }
   
   private double[][] add(double[][] A, double[][] B){
      double[][] C = new double[1][1];
   
      if(A.length == B.length && A[0].length == B[0].length){
         C = new double[A.length][A[0].length];
         for(int i=0;i<A.length;i++)
            for(int j=0;j<A[0].length;j++)
               C[i][j] = A[i][j] + B[i][j];  
      }
      
      return C;
   }
   
   
   private double[][] transpose(double[][] A){
      double[][] A_transposed = new double[A[0].length][A.length];
   
      for (int i = 0; i < A.length; i++)
         for (int j = 0; j < A[0].length; j++)
            A_transposed[j][i] = A[i][j];
   
      return A_transposed;
   }



   private double findMax(double[] data){
      double max = 0;
      double a = 0;
      double b = 0;
      int mid = 0;
      if(data.length>1){
         mid = data.length/2;
         double[] left = split(data, 0, mid);
         double[] right = split(data, mid, data.length);
         a = findMax(left);
         b = findMax(right);
         
         if(a>=b){
            max = a;
         }else{
            max = b;
         }
         
      }else{
         max = data[0];
      }
   
      return max;
   }

   private double findMin(double[] data){
      double min = 0;
      double a = 0;
      double b = 0;
      int mid = 0;
      if(data.length>1){
         mid = data.length/2;
         double[] left = split(data, 0, mid);
         double[] right = split(data, mid, data.length);
         a = findMin(left);
         b = findMin(right);
         
         if(a<=b){
            min = a;
         }else{
            min = b;
         }
      }else{
         min = data[0];
      }
   
      return min;
   }


   private double[] split(double[] data, int st, int end){
      double[] split_array = new double[end-st];
   
      for(int i=st;i<end;i++){
         split_array[i-st] = data[i];
      }
   
      return split_array;
   }

   private double[][] normalize(double[][] data){
      double max;
      double min;
      double rangeFloor = 0;
      double rangeCeiling = 1;
   
      double[][] normalized = new double[data.length][data[0].length];
      double[] dataTransposed = transpose(data)[0];
      max = findMax(dataTransposed);
      min = findMin(dataTransposed);
      
      for(int i=0;i<data.length;i++){
         for(int j=0;j<data[0].length;j++){
            normalized[i][j] = (((data[i][j]-min)/(max-min))*(rangeCeiling-rangeFloor))+rangeFloor;
         }
      }
   
      return normalized;
   }
   
   private void printMatrix(double[][] mtrx){
      for(int i=0;i<mtrx.length;i++){
         System.out.println();
         for(int j=0;j<mtrx[0].length;j++){
            System.out.print(mtrx[i][j]+", ");
         }
      }
      System.out.println();
   }
   
   private void printMatrix(int[][] mtrx){
      for(int i=0;i<mtrx.length;i++){
         System.out.println();
         for(int j=0;j<mtrx[0].length;j++){
            System.out.print(mtrx[i][j]+", ");
         }
      }
      System.out.println();
   }


}