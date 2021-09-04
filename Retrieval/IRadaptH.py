from AdaptionUtil import isCurrect,newGetComplain,isStaticAnalysisSuccess
import sys
from os.path import join
def adaptation(trainFocal,testFocal,trainAssert,testAssert,dataset):
    if dataset=="New":
        if trainFocal.endswith("null"):
            trainFocal = trainFocal.rstrip("null") + "{}"
        else:
            trainFocal = trainFocal
        if testFocal.endswith("null"):
            testFocal = testFocal.rstrip("null") + "{}"
        else:
            testFocal = testFocal
        trainFocal = "public class Main{" + trainFocal.replace('"<FocalMethod>"', "").replace(
            '"<AssertPlaceHolder>" ;', "") + "}"
        testFocal = "public class Main{" + testFocal.replace('"<FocalMethod>"', "").replace(
            '"<AssertPlaceHolder>" ;', "") + "}"
    else:
        if not isCurrect(trainFocal):
            return False, trainAssert
        if not isCurrect(testFocal):
            return False, trainAssert
        trainFocal = newGetComplain(trainFocal)
        testFocal = newGetComplain(testFocal)
        trainFocal = "public class Main{" + trainFocal + "}"
        testFocal = "public class Main{" + testFocal + "}"

    flag, adaptAssertion=isStaticAnalysisSuccess(trainFocal,testFocal,trainAssert,testAssert)

def ___main__():
    output_path = sys.argv[1]
    dataset = sys.argv[2]
    fRecordIRResult = open(join(output_path, "IRResultTest.txt"), 'r', encoding="utf-8")
    resultList = fRecordIRResult.read().rstrip("\n").split("\n")
    fRecordIRResult.close()
    lenNum = int(len(resultList) / 5) + 1
    tot = 0
    with open(join(output_path, "RAadapt-NN.txt"), "w+") as f:
        for i in range(0, lenNum):
            trainFocal = resultList[i * 5]
            testFocal = resultList[i * 5 + 1]
            trainAssert = resultList[i * 5 + 2].rstrip(" ")
            testAssert = resultList[i * 5 + 3].rstrip(" ")
            flag, adaptAssertion = adaptation(trainFocal, testFocal, trainAssert, testAssert, dataset)
            f.write(adaptAssertion+'\n')
            if flag:
                tot += 1
    print(tot * 1.0 / lenNum)