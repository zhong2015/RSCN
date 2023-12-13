package recommender.common;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;
import java.util.StringTokenizer;

public class CommonRec_RSCN {

    public static final int RMSE = 1;
    public static final int MAE = 2;
    public static final int R_Squared = 3;
    public static final int RMSLE = 4;
    public static final int MAPE = 5;
    public static final int RMSPE = 6;

    public static String dataSetName;
    public static String rsDataSetName;
    public static ArrayList<RTuple> trainDataSet =  null;
    public static ArrayList<RTuple> testDataSet =  null;

    public static int maxID = 0; //maxID=行数（user的个数）=列数（item的个数）【对称矩阵】

    public static double userRSetSize[];
    public static double itemRSetSize[];

    public static double lambda = 0.005; // 正则化参数
    public static double theta = 0.005; // 增广项参数
    public static double eta = 1;// 学习率参数
    public static int maxRound = 500; // 最多训练轮数
    public static int featureDimension = 0; // 特征维数
    public static double minGap = 0;

    public double  max_Res = -100;
    public double maxTotalTime = 0;
    public int max_Round = 0;

    public double  min_Error = 1e10; // 最小误差值
    public double cacheTotalTime = 0;
    public double minTotalTime = 0;
    public int min_Round = 0; // 记录达到最优结果时的最小迭代次数
    public int total_Round = 0;
    public double total_Time = 0;

    public static double[][] cachedP;
    public static double[][] cachedQ;
    public static double[][] cachedA;
    public static int mappingScale = 1000;
    public static double featureInitMax = 0.004;
    public static double featureInitScale = 0.004;

    // 辅助矩阵
    //特征矩阵
    public static double[][] A;
    //对应A，P和对应的拉格朗日乘子
    public double[][] P, Gamma_P;
    //对应A，Q和对应的拉格朗日乘子
    public double[][] Q, Gamma_Q;
    // 进行更新时的缓存矩阵
    public static double[] P_U, P_D, P_C, Q_U, Q_D, Q_C;

    // 存储特征值的路径
    public static String userPSaveDir;
    public static String itemQSaveDir;
    public static String ASaveDir;

    //Parallel Hyper-parameters
    public static int threadNum = 8;
    public static int blockSize = 0;
    public static int IDBlockSize = 0;
    public static int trainMaxID = 0;
    public static ArrayList<ArrayList<RTuple>> userMatrix;
    public static ArrayList<ArrayList<RTuple>> itemMatrix;
    public static ArrayList<ArrayList<RTuple>> trainUserNetblockMatrix;
    public static ArrayList<ArrayList<RTuple>> trainItemNetblockMatrix;
    public static int trainBlockSize = 0;

    public CommonRec_RSCN() {
        this.initInstanceFeatures();
    }

    /*
     * 初始化实例特征矩阵
     */
    public void initInstanceFeatures() {
        // 加1是为了在序号上与ID保持一致
        P = new double[maxID + 1][featureDimension];
        Q = new double[maxID + 1][featureDimension];
        A = new double[maxID + 1][featureDimension];
        Gamma_P = new double[maxID + 1][featureDimension];
        Gamma_Q = new double[maxID + 1][featureDimension];

        for (int i = 1; i <= maxID; i++) {
            for (int j = 0; j < featureDimension; j++) {

                P[i][j] = cachedP[i][j];
                Q[i][j] = cachedQ[i][j];
                A[i][j] = cachedA[i][j];
                Gamma_P[i][j] = 0;
                Gamma_Q[i][j] = 0;
            }
        }
    }

    /*
     * 生成初始的训练集、测试集以及统计各个结点的评分数目
     */
    public static void dataLoad(String trainFileName, String testFileName, String separator) throws IOException {

        //生成初始的训练集
        trainDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, trainFileName, trainDataSet, 1);

        //生成初始的验证集or测试集
        testDataSet = new ArrayList<RTuple>();
        dataSetGenerator(separator, testFileName, testDataSet, 2);

        initRatingSetSize();
    }

    /*
     * 数据集生成器
     */
    public static void dataSetGenerator(String separator, String fileName, ArrayList<RTuple> dataSet, int flag) throws IOException {

        File fileSource = new File(fileName);
        BufferedReader in = new BufferedReader(new FileReader(fileSource));

        String line;
        while (((line = in.readLine()) != null)){
            StringTokenizer st = new StringTokenizer(line, separator);
            String personID = null;
            if (st.hasMoreTokens())
                personID = st.nextToken();
            String movieID = null;
            if (st.hasMoreTokens())
                movieID = st.nextToken();
            String personRating = null;
            if (st.hasMoreTokens())
                personRating = st.nextToken();
            int iUserID = Integer.valueOf(personID);
            int iItemID = Integer.valueOf(movieID);

            // 记录下最大的itemid和userid；因为itemid和userid是连续的，所以最大的itemid和userid也代表了各自的数目
            maxID = (maxID > iUserID) ? maxID : iUserID;
            maxID = (maxID > iItemID) ? maxID : iItemID;
            double dRating = Double.valueOf(personRating);

            RTuple temp1 = new RTuple();
            temp1.userID = iUserID;
            temp1.itemID = iItemID;
            temp1.rating = dRating;
            dataSet.add(temp1);

            if(iUserID != iItemID){
               // 不是对角线元素的话，则读入对称项
                RTuple temp2 = new RTuple();
                temp2.userID = iItemID;
                temp2.itemID = iUserID;
                temp2.rating = dRating;
                dataSet.add(temp2);
            }

            // Parallel
            if(flag == 1){
                trainMaxID = (trainMaxID > iUserID) ? trainMaxID : iUserID;
                trainMaxID = (trainMaxID > iItemID) ? trainMaxID : iItemID;
            }
        }
        in.close();
    }

    //Parallel_初始化训练矩阵-以user为索引
    public static void initNetBlockMatrix(){

        //初始化训练集的评分矩阵-以user为索引
        userMatrix = new ArrayList<ArrayList<RTuple>>();
        itemMatrix = new ArrayList<ArrayList<RTuple>>();

        for(int i = 0; i <= trainMaxID; i++){
            ArrayList<RTuple> tmpUList = new ArrayList<RTuple>();
            userMatrix.add(tmpUList);
            
            ArrayList<RTuple> tmpIList = new ArrayList<RTuple>();
            itemMatrix.add(tmpIList);
        }

        for(RTuple trainR: trainDataSet){
            userMatrix.get(trainR.userID).add(trainR);
            itemMatrix.get(trainR.itemID).add(trainR);
        }

        trainUserNetblockMatrix = new ArrayList<ArrayList<RTuple>>();
        trainItemNetblockMatrix = new ArrayList<ArrayList<RTuple>>();
        
        //初始化
        for(int i = 0; i < threadNum; i++){
        	ArrayList<RTuple> tmpUList = new ArrayList<RTuple>();
        	trainUserNetblockMatrix.add(tmpUList);
        	
        	ArrayList<RTuple> tmpIList = new ArrayList<RTuple>();
        	trainItemNetblockMatrix.add(tmpIList);
        }
        //添加数据
        for(int i = 0; i < threadNum; i++){
            int k = (i + 1) * trainBlockSize;

            if(i == threadNum - 1){
                k = trainMaxID + 1;
            }

            for(int j = i * trainBlockSize; j < k; j++){
                ArrayList<RTuple> tmpUList = userMatrix.get(j);
                ArrayList<RTuple> tmpIList = itemMatrix.get(j);
                for (RTuple trainR:tmpUList){
                    trainUserNetblockMatrix.get(i).add(trainR);
                }
                for (RTuple trainR:tmpIList){
                    trainItemNetblockMatrix.get(i).add(trainR);
                }
            }
        }
    }

    /*
     * 统计各个结点的评分数目
     */
    public static void initRatingSetSize() {
        userRSetSize = new double[maxID + 1];
        itemRSetSize = new double[maxID + 1];

        for (int i = 1; i <= maxID; i++) {
            userRSetSize[i] = 0;
            itemRSetSize[i] = 0;
        }

        for (RTuple tempRating : trainDataSet) {
            userRSetSize[tempRating.userID] += 1;
            itemRSetSize[tempRating.itemID] += 1;
        }
    }


    /*
     * 声明辅助矩阵，并用随机数进行初始化
     */
    public static void initStaticFeatures() throws IOException {

        // 加1是为了在序号上与ID保持一致
        cachedP =  new double[maxID + 1][featureDimension];
        cachedQ =  new double[maxID + 1][featureDimension];
        cachedA =  new double[maxID + 1][featureDimension];

        userPSaveDir = userPSaveDir +  featureDimension + ".txt";
        itemQSaveDir = itemQSaveDir + featureDimension + ".txt";
        ASaveDir = ASaveDir + featureDimension + ".txt";

        File userXFile = new File(userPSaveDir);        // new File(".") 表示用当前路径 生成一个File实例!!!并不是表达创建一个 . 文件
        File itemYFile = new File(itemQSaveDir);
        File AFile = new File(ASaveDir);

        if(userXFile.exists() && itemYFile.exists() && AFile.exists()) { // 如果由指定路径下的文件或目录存在则返回 TRUE，否则返回 FALSE。
            System.out.println("准备读取指定初始值...");
            readFeatures(cachedP, userPSaveDir);  // 读取user特征矩阵
            readFeatures(cachedQ, itemQSaveDir);  // 读取user特征矩阵
            readFeatures(cachedA, ASaveDir);  // 读取user特征矩阵
            System.out.println("读取完毕！！！");
        }else{
            System.out.println("准备生成随机初始值...");
            // 初始化特征矩阵,采用随机值,从而形成一个K阶逼近
            Random random = new Random(System.currentTimeMillis());
            for (int i = 1; i <= maxID; i++) {
                // 特征矩阵的初始值在(0,0.004]
                for (int j = 0; j < featureDimension; j++) {
                    int tempP = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    int tempQ = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    int tempA = random.nextInt(mappingScale); //返回[0,mappingScale)随机整数
                    cachedP[i][j] = featureInitMax - featureInitScale * tempP / mappingScale;
                    cachedQ[i][j] = featureInitMax - featureInitScale * tempQ / mappingScale;
                    cachedA[i][j] = featureInitMax - featureInitScale * tempA / mappingScale;
                }
            }

            // 写入文件
            writeFeatures(cachedP,userPSaveDir);
            writeFeatures(cachedQ,itemQSaveDir);
            writeFeatures(cachedA,ASaveDir);
            System.out.println("写入随机初始值完毕！！！");
        }

        // 声明辅助矩阵
        initAuxArray();
    }

    private static void writeFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {

        FileWriter fw = new FileWriter(featureSaveDir);

        for(int i = 1; i <= maxID; i++) {
            for(int k = 0; k < featureDimension; k++) {
                fw.write(catchedFeatureMatrix[i][k] + "::");
            }
            fw.write("\n");
        }
        fw.flush();
        fw.close();
    }

    private static void readFeatures(double[][] catchedFeatureMatrix, String featureSaveDir) throws IOException {

        BufferedReader in = new BufferedReader(new FileReader(featureSaveDir));
        String line;  // 一行数据
        int i = 1;    // 行标
        while((line = in.readLine()) != null){
            String [] temp = line.split("::"); // 各数字之间用"::"间隔
            for(int k = 0; k < featureDimension; k++) {
                catchedFeatureMatrix[i][k] = Double.valueOf(temp[k]);
            }
            i++;
        }
        in.close();
    }

    /*
     * 声明辅助矩阵
     */
    public static void initAuxArray() {

        // 加1是为了在序号上与ID保持一致
        P_U = new double[maxID + 1];
        P_D = new double[maxID + 1];
        P_C = new double[maxID + 1];
        Q_U = new double[maxID + 1];
        Q_D = new double[maxID + 1];
        Q_C = new double[maxID + 1];
    }

    /*
     * 将辅助矩阵的元素置为0
     */
    public void resetAuxArray() {
        for (int i = 1; i <= maxID; i++) {

            P_U[i] = 0;
            P_D[i] = 0;
            Q_U[i] = 0;
            Q_D[i] = 0;
        }
    }

    // 计算两个向量点乘
    public static double dotMultiply(double[] x, double[] y) {
        double sum = 0;
        for (int i = 0; i < x.length; i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }

    /*
     * 计算考虑线性偏差的预测值
     */
    public double getPrediction(int firstID, int secondID) {
        double ratingHat = 0;
        ratingHat += dotMultiply(A[firstID], A[secondID]);
        return ratingHat;
    }

    public double testRMSE() {

        // 计算在测试集上的RMSE
        double sumRMSE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumRMSE += Math.pow((actualRating - ratinghat), 2);
            sumCount++;
        }
        double RMSE = Math.sqrt(sumRMSE / sumCount);
        return RMSE;
    }

    public double testMAE() {
        // 计算在测试集上的MAE
        double sumMAE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumMAE += Math.abs(actualRating - ratinghat);
            sumCount++;
        }
        double MAE = sumMAE / sumCount;
        return MAE;
    }

    public double avg_testData(){
        double sum = 0, count = 0;
        for (RTuple testR : testDataSet) {
            sum += testR.rating;
            count++;
        }
        double result = sum / count;
        return result;
    }
    public double testR_Square(double avg){

        double SSR = 0, SST = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            SSR += Math.pow((actualRating - ratinghat), 2);
            SST += Math.pow((actualRating - avg), 2);
        }
        double result = 1 - SSR / SST;
        return result;
    }

    public double testRMSLE() {
        // 计算在测试集上的MAE
        double sumRMSLE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumRMSLE += Math.pow((Math.log(actualRating + 1) - Math.log(ratinghat + 1)), 2);
            sumCount++;
        }
        double RMSLE = sumRMSLE / sumCount;
        return RMSLE;
    }

    public double testMAPE() {
        // 计算在测试集上的MAE
        double sumMAPE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumMAPE += Math.abs((actualRating - ratinghat) / actualRating);
            sumCount++;
        }
        double MAPE = sumMAPE / sumCount;
        return MAPE;
    }

    public double testRMSPE() {
        // 计算在测试集上的MAE
        double sumRMSPE = 0, sumCount = 0;
        for (RTuple testR : testDataSet) {
            double actualRating = testR.rating;
            double ratinghat = getPrediction(testR.userID, testR.itemID);
            sumRMSPE += Math.pow(((actualRating - ratinghat) / actualRating), 2);
            sumCount++;
        }
        double RMSPE = sumRMSPE / sumCount;
        return RMSPE;
    }

    public void outPutModelSym() throws IOException {

    	FileWriter fwUp = new FileWriter(
                new File("./" + rsDataSetName + "_ModelSym_Up.txt"), true);
        FileWriter fwDown = new FileWriter(
                new File("./" + rsDataSetName + "_ModelSym_Down.txt"), true);
        fwUp.write("i-j:\n");
        fwDown.write("j-i:\n");
        for(int i = 1; i <= 1500; i++){
            for(int j = 1500; j >= i; j--){
                double ratinghatUp = getPrediction(i, j);
                double ratinghatDown = getPrediction(j, i);
                fwUp.write(ratinghatUp + "\n");
                fwDown.write(ratinghatDown + "\n");
                fwUp.flush();
                fwDown.flush();
            }
        }
        fwUp.close();
        fwDown.close();
    }

    public void outPutLFMatrices() throws IOException {

        FileWriter P_fw = new FileWriter(
                new File("./" + rsDataSetName + "_P.txt"), true);
        FileWriter Q_fw = new FileWriter(
                new File("./" + rsDataSetName + "_Q.txt"), true);
        FileWriter A_fw = new FileWriter(
                new File("./" + rsDataSetName + "_A.txt"), true);

        for(int id = 1; id <= maxID; id++) {
            for(int dim = 0; dim < featureDimension; dim++){
                P_fw.write(P[id][dim] + "\n");
                P_fw.flush();
                Q_fw.write(Q[id][dim] + "\n");
                Q_fw.flush();
                A_fw.write(A[id][dim] + "\n");
                A_fw.flush();
            }
        }
        P_fw.close();
        Q_fw.close();
        A_fw.close();
    }

    public void outPutModelNonnega() throws IOException {

        FileWriter fw = new FileWriter(
                new File("./" + rsDataSetName + "_ModelNonnega.txt"), true);
        for(int id = 1; id <= maxID; id++) {
            for(int dim = 0; dim < featureDimension; dim++){
                fw.write(A[id][dim] + "\n");
                fw.flush();
            }
        }
        fw.close();
    }
}
