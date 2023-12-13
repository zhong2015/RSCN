package recommender;

import recommender.common.CommonRec_RSCN;
import recommender.common.RTuple;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class RSCN extends CommonRec_RSCN {

    public RSCN() {
        super();
    }

    public static void main(String[] args) throws IOException {

        int[] rsArr = new int[]{1};
        for(int rs:rsArr) {
            CommonRec_RSCN.dataSetName = "zyr_162425";
            CommonRec_RSCN.rsDataSetName = String.valueOf(rs) + "_" + CommonRec_RSCN.dataSetName;
            String filePath = "E:\\workspace-jasonchung\\Selected_DS\\7-1-2\\";
            CommonRec_RSCN.dataLoad(filePath + CommonRec_RSCN.rsDataSetName + "_train.txt", filePath + CommonRec_RSCN.rsDataSetName + "_test.txt", "::");
            System.out.println("当前物种的蛋白质总数：" + maxID);
            System.out.println("训练集的容量：" + CommonRec_RSCN.trainDataSet.size());
            System.out.println("测试集的容量：" + CommonRec_RSCN.testDataSet.size());

            // 设置公共参数
            CommonRec_RSCN.maxRound = 1000;
            CommonRec_RSCN.minGap = 1e-5;

            for(int tempdim = 20; tempdim <= 20; tempdim += CommonRec_RSCN.featureDimension){
                CommonRec_RSCN.featureDimension = tempdim;
                CommonRec_RSCN.userPSaveDir = "./savedLFs/"+ CommonRec_RSCN.dataSetName +"/P";
                CommonRec_RSCN.itemQSaveDir = "./savedLFs/"+ CommonRec_RSCN.dataSetName +"/Q";
                CommonRec_RSCN.ASaveDir = "./savedLFs/"+ CommonRec_RSCN.dataSetName +"/A";

                // 初始化特征矩阵
                CommonRec_RSCN.initStaticFeatures();

                experimenter(CommonRec_RSCN.RMSE);
//                    experimenter(CommonRec_ADMM.MAE);
            }
        }
    }

    /*
     * 综合实验
     */
    public static void experimenter(int metrics) throws IOException {
        long file_tMills = System.currentTimeMillis(); //用于给train函数打开在当前函数所创建的文件
        FileWriter fw;
        if(metrics == CommonRec_RSCN.RMSE)
            fw = new FileWriter(new File("./" + rsDataSetName + "_RMSE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);
        else
            fw = new FileWriter(new File("./" + rsDataSetName + "_MAE_" +
                    Thread.currentThread().getStackTrace()[1].getClassName().trim() + "_" + file_tMills +
                    "dim=" + featureDimension + ".txt"), true);

        String blankStr = "                          ";
        String starStr = "****************************************************************";
        String equalStr = "=====================================";
        String formStr = "&&&&&&&&&&&&&&&&&&&&&&";

        // 按正则项系数lambda的不同取值开始测试
        for (double tempLam = Math.pow(2,-7); tempLam >= Math.pow(2,-7); tempLam *= Math.pow(2,-1)) {

            CommonRec_RSCN.lambda = tempLam;

            // 打印标题项
            System.out.println("\n" + starStr);
            System.out.println(blankStr + "featureDimension——>" + CommonRec_RSCN.featureDimension);
            System.out.println(blankStr + "lambda——>" + CommonRec_RSCN.lambda);
            System.out.println(blankStr + "minGap——>" + CommonRec_RSCN.minGap);
            System.out.println(starStr);

            fw.write("\n" + starStr + "\n");
            fw.write(blankStr + "featureDimension——>" + CommonRec_RSCN.featureDimension + "\n");
            fw.write(blankStr + "lambda——>" + CommonRec_RSCN.lambda + "\n");
            fw.write(blankStr + "minGap——>" + CommonRec_RSCN.minGap + "\n");
            fw.write(starStr + "\n");
            fw.flush();

            for(double tempTheta = Math.pow(2,-5); tempTheta >= Math.pow(2,-5); tempTheta -= 0.2){

                CommonRec_RSCN.theta = tempTheta;

                // 打印标题项
                System.out.println("\n" + equalStr);
                System.out.println("        theta——>"  + CommonRec_RSCN.theta);
                System.out.println(equalStr);

                fw.write("\n" + equalStr + "\n");
                fw.write("        theta——>" + CommonRec_RSCN.theta + "\n");
                fw.write(equalStr + "\n");
                fw.flush();

                // 按学习率eta取值的不同进行测试
                for(double tempEta = Math.pow(2,-3); tempEta <= Math.pow(2,-3); tempEta *= 2){

                    CommonRec_RSCN.eta = tempEta;

                    fw.write("\n" + formStr + "\n");
                    fw.write("    Eta——>" + CommonRec_RSCN.eta + "\n");
                    fw.write(formStr + "\n");
                    fw.flush();

                    System.out.println("\n" + formStr);
                    System.out.println("    Eta——>" + CommonRec_RSCN.eta);
                    System.out.println(formStr);

                    // 为确保每一次gamma取新值后是一个新的更新过程，则每次重新创建一个MSNLF对象，这些对象的参数取值是一致的
                    RSCN trainANLF = new RSCN();
                    // 开始训练
                    trainANLF.train(metrics, fw);

                    // 输出最优值信息
                    System.out.println("Min training Error:\t\t\t" + trainANLF.min_Error);
                    System.out.println("Min total training epochs:\t\t" + trainANLF.min_Round);
                    System.out.println("Total Round:\t\t" + trainANLF.total_Round);
                    System.out.println("Min total training time:\t\t" + trainANLF.minTotalTime);
                    System.out.println("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round);
                    System.out.println("Total training time:\t\t" + trainANLF.total_Time);
                    System.out.println("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round);

                    fw.write("Min training Error:\t\t\t" + trainANLF.min_Error + "\n");
                    fw.write("Min total training epochs:\t\t" + trainANLF.min_Round + "\n");
                    fw.write("Total Round:\t\t" + trainANLF.total_Round + "\n");
                    fw.write("Min total training time:\t\t" + trainANLF.minTotalTime + "\n");
                    fw.write("Min average training time:\t\t" + trainANLF.minTotalTime / trainANLF.min_Round + "\n");
                    fw.write("Total training time:\t\t" + trainANLF.total_Time + "\n");
                    fw.write("Average training time:\t\t" + trainANLF.total_Time / trainANLF.total_Round + "\n");
                    fw.flush();
                }
            }
        }
        fw.close();
    }

    public void train(int metrics, FileWriter fw) throws IOException {

        double lastErr = 0;

        // 初始化：将所有的rating估计值缓存起来，提高计算效率
        for (RTuple trainR : trainDataSet) {
            double ratingHat = dotMultiply(P[trainR.userID], Q[trainR.itemID]);
            trainR.ratingHat = ratingHat;
        }

        for (int round = 1; round <= maxRound; round++) {

            double startTime = System.currentTimeMillis();
            double rho;
            // 在第K列上进行每一轮轮迭代，更新P矩阵
            for (int dim = 0; dim < featureDimension; dim++) {
                // 将相关的辅助变量置为0
                resetAuxArray();

                for (RTuple trainR : trainDataSet) {

                    double PQ_notk = trainR.ratingHat - P[trainR.userID][dim] * Q[trainR.itemID][dim];
                    P_U[trainR.userID] += Q[trainR.itemID][dim] * (trainR.rating - PQ_notk);
                    P_D[trainR.userID] += Q[trainR.itemID][dim] * Q[trainR.itemID][dim] + lambda;

                    Q_U[trainR.itemID] += P[trainR.userID][dim] * (trainR.rating - PQ_notk);
                    Q_D[trainR.itemID] += P[trainR.userID][dim] * P[trainR.userID][dim] + lambda;
                }

                for (int id = 1; id <= maxID; id++){

                    P_C[id] = P[id][dim];
                    Q_C[id] = Q[id][dim];

                    rho = theta * userRSetSize[id];
                    
                    P_U[id] += rho * A[id][dim];
                    P_U[id] -= Gamma_P[id][dim];
                    P_D[id] += rho;
                    P[id][dim] = P_U[id] / P_D[id];

                    Q_U[id] += rho * A[id][dim];
                    Q_U[id] -= Gamma_Q[id][dim];
                    Q_D[id] += rho;
                    Q[id][dim] = Q_U[id] / Q_D[id];

                    double tempA = (P[id][dim] + Q[id][dim]) * 0.5 + (Gamma_P[id][dim] + Gamma_Q[id][dim]) / (2 * rho);
                    if(tempA > 0)
                        A[id][dim] = tempA;
                    else
                        A[id][dim] = 0;

                    Gamma_P[id][dim] += eta * rho * (P[id][dim] - A[id][dim]);
                    Gamma_Q[id][dim] += eta * rho * (Q[id][dim] - A[id][dim]);
                }

                for (RTuple trainR : trainDataSet) {
                    // 根据最新的X和Y来对所有的rating估计值进行更新，因为当前X和Y的第k列更新了
                    double ratingHatNew = P[trainR.userID][dim] * Q[trainR.itemID][dim]
                            - P_C[trainR.userID] * Q_C[trainR.itemID];
                    trainR.ratingHat = trainR.ratingHat + ratingHatNew;
                }
            }

            double endTime = System.currentTimeMillis();
            cacheTotalTime += endTime - startTime;
            total_Time += endTime - startTime;

            // 计算本轮训练结束后，在测试集上的误差
            double curErr;
            if (metrics == CommonRec_RSCN.RMSE) {
                curErr = testRMSE();
            } else {
                curErr = testMAE();
            }
            fw.write(curErr + "\n");
            fw.flush();
            System.out.println(curErr);
            System.out.println(endTime - startTime);

            total_Round += 1;
            if (min_Error > curErr) {
                min_Error = curErr;
                min_Round = round;
                minTotalTime += cacheTotalTime;
                cacheTotalTime = 0;
            }

            if (Math.abs(curErr - lastErr) > minGap)
                lastErr = curErr;
            else
                break;
        }
//        outPutModelSym();
//        outPutLFMatrices();
    }
}
