public class Deep_Neural_Network_Test{

   static Deep_Neural_Network n;
   static double[][][] braille_matrix = {{{1, 0, 0, 0, 0, 0}}, {{1, 1, 0, 0, 0, 0}}, {{1, 0, 0, 1, 0, 0}}, {{1, 0, 0, 1, 1, 0}}, {{1, 0, 0, 0, 1, 0}},
                            {{1, 1, 0, 1, 0, 0}}, {{1, 1, 0, 1, 1, 0}}, {{1, 1, 0, 0, 1, 0}}, {{0, 1, 0, 1, 0, 0}}, {{0, 1, 0, 1, 1, 0}},
                            {{1, 0, 1, 0, 0, 0}}, {{1, 1, 1, 0, 0, 0}}, {{1, 0, 1, 1, 0, 0}}, {{1, 0, 1, 1, 1, 0}}, {{1, 0, 1, 0, 1, 0}},
                            {{1, 1, 1, 1, 0, 0}}, {{1, 1, 1, 1, 1, 0}}, {{1, 1, 1, 0, 1, 0}}, {{0, 1, 1, 1, 0, 0}}, {{0, 1, 1, 1, 1, 0}},
                            {{1, 0, 1, 0, 0, 1}}, {{1, 1, 1, 0, 0, 1}}, {{0, 1, 0, 1, 1, 1}}, {{1, 0, 1, 1, 0, 1}}, {{1, 0, 1, 1, 1, 1}},
                            {{1, 0, 1, 0, 1, 1}}};
                            
                            
   static double[][][][] X_data = {{{{1, 0, 0, 0, 0, 0}}, {{1, 1, 0, 0, 0, 0}}, {{1, 0, 0, 1, 0, 0}}, {{1, 0, 0, 1, 1, 0}}, {{1, 0, 0, 0, 1, 0}},
                            {{1, 1, 0, 1, 0, 0}}, {{1, 1, 0, 1, 1, 0}}, {{1, 1, 0, 0, 1, 0}}, {{0, 1, 0, 1, 0, 0}}, {{0, 1, 0, 1, 1, 0}},
                            {{1, 0, 1, 0, 0, 0}}, {{1, 1, 1, 0, 0, 0}}, {{1, 0, 1, 1, 0, 0}}},
                            {{{1, 0, 1, 1, 1, 0}}, {{1, 0, 1, 0, 1, 0}},
                            {{1, 1, 1, 1, 0, 0}}, {{1, 1, 1, 1, 1, 0}}, {{1, 1, 1, 0, 1, 0}}, {{0, 1, 1, 1, 0, 0}}, {{0, 1, 1, 1, 1, 0}},
                            {{1, 0, 1, 0, 0, 1}}, {{1, 1, 1, 0, 0, 1}}, {{0, 1, 0, 1, 1, 1}}, {{1, 0, 1, 1, 0, 1}}, {{1, 0, 1, 1, 1, 1}},
                            {{1, 0, 1, 0, 1, 1}}}};
   static double[][][][] Y_data = new double[2][13][1][2];
 
   static double[][][] X_1;
   static double[][][] Y_1;

   static double[][][] X_2;
   static double[][][] Y_2;

   static double[] labels = {1, 1};

   static double[][] input_data;
   static double[][] output_data;

   public static void main(String[] args){
      n = new Deep_Neural_Network(3, 6, 6, 2);
      
      labelData();
      separateData();
      
      
      
      
      
      double[][][] training_dataset = X_1;
        //a e o x y
      double[][][] training_dataset_targets = Y_1;
      
   
      input_data = braille_matrix[7];
      
      //display();
      int iterations = 10;
      for(int t=0;t<iterations;t++){
         System.out.println("TRAINING ITERATION "+t);
      
         double[][][] tD = shuffle(training_dataset, training_dataset_targets);
         training_dataset = split(tD, 0, training_dataset.length);
         training_dataset_targets = split(tD, training_dataset.length, (training_dataset.length*2));
      
         for(int i=0;i<13;i++){
            n.train(transpose(training_dataset[i]), transpose(training_dataset_targets[i]));
         }
         
         
      }
      
      display();
   }
   
   
   public static double[][][] shuffle(double[][][] data, double[][][] data_targets){
      double[][][] new_data = new double[data.length*2][data[0].length][data[0][0].length];
      int[] pickedIndexes = new int[data.length];
   
      for(int i=0;i<data.length;i++){
         int randomIndex = getRandomIndex(pickedIndexes);           
         new_data[i] = data[randomIndex];
         new_data[i+data.length] = data_targets[i];
         pickedIndexes[randomIndex] = 1;
      }
   
      return new_data;
   }
   
   private static int getRandomIndex(int[] picked){
      int counter = 0;
      //System.out.println("Picked "+picked.length);
      int randomIndex = (int) (Math.random()*picked.length);
      //System.out.println("Random "+randomIndex);
      for(int r=0;r<picked.length;r++){
         if(picked[randomIndex] == 1){
            counter += 1;
            //System.out.println("Checked and failed "+counter);
            if(counter == picked.length){
               randomIndex = getRandomIndex(picked);
            }
         }
      }
      return randomIndex;
   }
   
   private static double[][][] split(double[][][] data, int begin, int end){
      int size = end-begin;
      double[][][] splitData = new double[size][data.length][data[0].length];
   
      for(int i=begin;i<end;i++){
         if(begin<splitData.length){
            splitData[i] = data[i];
         }else{
            splitData[i-size] = data[i];
         }
      }
      
      return splitData;
   }
   
   
   public static void display(){
      output_data = n.query(transpose(input_data));
      System.out.println("input data");
      printMatrix(input_data);
      System.out.println("output data");
      printMatrix(output_data);
   }
   
   public static void printMatrix(double[][] mtrx){
   
      for(int i=0;i<mtrx.length;i++){
         System.out.println();
         for(int j=0;j<mtrx[0].length;j++){
            System.out.print(mtrx[i][j]+", ");
         }
      }
      System.out.println();
   }
     
   
   static private double[][] transpose(double[][] A){
      double[][] A_transposed = new double[A[0].length][A.length];
   
      for (int i = 0; i < A.length; i++)
         for (int j = 0; j < A[0].length; j++)
            A_transposed[j][i] = A[i][j];
   
      return A_transposed;
   }
   
   
   
   

   
   
   static void labelData(){
      for(int s=0;s<X_data.length;s++){
         for(int a=0;a<X_data[s].length;a++){
            Y_data[s][a][0][0] = 0;
            Y_data[s][a][0][1] = 0;
         
            Y_data[s][a][0][s] = labels[s];
            
         }
      }
   }
   
   static void separateData(){
      int setLength = (X_data[0].length/2);
      System.out.println("setLength: "+setLength);
      X_1 = new double[X_data[0].length][X_data[0].length][X_data[0][0].length];
      Y_1 = new double[X_data[0].length][Y_data[0].length][Y_data[0][0].length];
      
      X_2 = X_1;
      Y_2 = Y_1;
      
   
      for(int x=0;x<=setLength;x++){
         X_1[x] = X_data[0][x];
         Y_1[x] = Y_data[0][x];
         X_2[x] = X_data[1][x];
         Y_2[x] = Y_data[1][x];
            
         X_1[setLength+x] = X_data[1][setLength+x];
         Y_1[setLength+x] = Y_data[1][setLength+x];
         X_2[setLength+x] = X_data[0][setLength+x];
         Y_2[setLength+x] = Y_data[0][setLength+x];
      }
      
            
      
   
   }
   
   
   
   
   
   
   
   
   
   
   
}