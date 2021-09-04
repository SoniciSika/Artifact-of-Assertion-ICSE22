from javalang.tree import MethodInvocation
from javalang.tree import VariableDeclarator
from javalang.tree import ConstructorDeclaration
from javalang.tree import MemberReference
from javalang.tree import Literal
import math
import javalang

listKeyWordsVarAndMethod = ["assertEquals", "org.junit.Assert", "assertTrue", "assertFalse",
                            "assertNotNull", "assertThat", "assertNull", "assertArrayEquals", "assertSame"]

class Stack(object):
    def __init__(self):
         self.items = []

    def is_empty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)
def isCurrect(content):
    stack=Stack()
    stack.push("{")
    content=content.split(" ")
    for i in range(4,len(content)):
        if content[i]=="{":
            stack.push("{")
        elif content[i]=="}":
            if stack.peek()=="{":
                stack.pop()
        if stack.size()==0:
            return True
    return False
def getContentComplain(content,posSecondFunc):
    contentList = content.split(" ")
    startPos = 0
    endPos = 0
    topend = 0
    if contentList[2] != ')':
        for i in range(2, len(contentList)):
            if contentList[i] == ')':
                topend = i
                break
        newList = getNewList(contentList, 0, topend + 1)
        newList.extend(contentList[topend + 1:])
        contentList = newList
    if posSecondFunc!=0:
        startPos = posSecondFunc
        for i in range(posSecondFunc, len(contentList)):
            if contentList[i] == ')' and contentList[i-1] != ')'and contentList[i + 1] == '{':
                endPos = i
                break
    newFinal = getNewList(contentList, startPos, endPos + 1)
    newFinal.extend(contentList[endPos + 1:])
    tempA=contentList[:startPos]
    tempA.extend(newFinal)
    newFinal=tempA
    outString = newFinal[0]
    for l in range(1, len(newFinal)):
        outString = outString + " "
        outString = outString + newFinal[l]
    outString = outString + "\n"
    return outString
def getFuncStart(content):
    isStart=False
    stack=Stack()
    content=content.split(" ")
    for i in range(0,len(content)):
        if content[i]=="{":
            isStart=True
            stack.push("{")
        elif content[i]=="}":
            if stack.peek()=="{":
                stack.pop()
        if stack.size()==0 and isStart:
            if i+1!=len(content):
                return i+1
            else:
                return 0
def getNewList(contentL,startP,endP):
    tempList = contentL[startP:endP]
    z = 0
    if endP-startP==3:
        return tempList
    else:
        for l in tempList:
            if l==',':
                z+=1
    newList = []
    if z==0:
        for l in tempList:
            if l == ')':
                newList.append("var")
                newList.append(")")
            else:
                newList.append(l)
    else:
        isHavePara=False
        for l in tempList:
            if l == ',':
                isHavePara=True
                newList.append("var")
                newList.append(",")
            elif l==')':
                if isHavePara:
                    newList.append("var")
                newList.append(")")
            else:
                newList.append(l)
    return newList
def newGetComplain(content):
    pos=getFuncStart(content)
    outString=getContentComplain(content,pos)
    return outString

def getHump(input):
    for i in range(0, len(input) - 1):
        if input[i] > 'a' and input[i] < 'z' and input[i + 1] > 'A' and input[i + 1] < 'Z':
            return input[:i + 1] + " " + input[i + 1:]
    return input


def getPostion(content):
    listContent = javalang.tokenizer.tokenize(content)
    mapTemp = {}
    for l in listContent:
        if l.value not in mapTemp.keys():
            mapTemp[l.value] = l.position[1]
    return mapTemp


def getListType(tree, methodName):
    ListMethod = []
    ListVariable = []
    ListLiteral = []
    ListOther = []
    for path, node in tree:
        children = node
        if isinstance(children, MethodInvocation):
            if children.member not in listKeyWordsVarAndMethod:
                ListMethod.append(children.member)
            if children.qualifier not in listKeyWordsVarAndMethod:
                ListVariable.append(children.qualifier)
        if isinstance(children, ConstructorDeclaration):
            if methodName != children.name:
                ListMethod.append(children.name)
        if isinstance(children, Literal):
            if children.value != '"<AssertPlaceHolder>"':
                ListLiteral.append(children.value)
        if isinstance(children, MemberReference):
            ListVariable.append(children.qualifier)
            ListVariable.append(children.member)
        if isinstance(children, VariableDeclarator):
            ListVariable.append(children.name)
        if not (isinstance(children, MethodInvocation) or isinstance(children, ConstructorDeclaration) or isinstance(
                children, Literal) or
                isinstance(children, MemberReference) or isinstance(children, VariableDeclarator)):
            if str(children) not in listKeyWordsVarAndMethod:
                ListOther.append(str(children))
    rListMethod = []
    for l in ListMethod:
        if l not in rListMethod and l != "":
            rListMethod.append(l)
    rListVariable = []
    for l in ListVariable:
        if l not in rListVariable and l != "":
            rListVariable.append(l)
    rListLiteral = []
    for l in ListLiteral:
        if l not in rListLiteral:
            rListLiteral.append(l)
    rListOther = []
    for l in ListOther:
        if l not in rListOther:
            rListOther.append(l)
    return rListMethod, rListVariable, rListLiteral, rListOther


def isStaticAnalysisSuccess(trainFocal, testFocal, trainAssert, testAssert):
    try:
        treeTrainFocal = javalang.parse.parse(trainFocal)
    except:
        return False, trainAssert
    methodInvoTrainFocal, variableTrainFocal, literalOtherTrainFocal, otherTrainFocal = getListType(treeTrainFocal, trainFocal.split(" ")[0])
    mapRecordCol = getPostion(trainFocal)
    try:
        treeTestFocal = javalang.parse.parse(testFocal)
    except:
        return False, trainAssert
    methodInvoTestFocal, variableTestFocal, literalTestFocal, otherTestFocal = getListType(treeTestFocal, testFocal.split(" ")[0])
    mapRecordCol1 = getPostion(testFocal)
    mapAndList = {}
    for l in methodInvoTrainFocal:
        if l not in methodInvoTestFocal:
            tempList = []
            for l1 in methodInvoTestFocal:
                tempList.append(l1)
            mapAndList[l] = tempList
    for l in variableTrainFocal:
        if l not in variableTestFocal:
            tempList = []
            for l1 in variableTestFocal:
                tempList.append(l1)
            mapAndList[l] = tempList
    for l in literalOtherTrainFocal:
        if l not in literalTestFocal:
            tempList = []
            for l1 in literalTestFocal:
                tempList.append(l1)
            mapAndList[l] = tempList
    for l in otherTrainFocal:
        if l not in otherTestFocal:
            tempList = []
            for l1 in otherTestFocal:
                tempList.append(l1)
            mapAndList[l] = tempList
    ListOriginal = testFocal.split(" ")
    isPos = 0
    for i in range(0, len(trainAssert)):
        if trainAssert[i] == "(":
            isPos = i
            break
    tempTrainAssert = trainAssert[isPos + 2:len(trainAssert) - 2]
    tempTrainAssertList = tempTrainAssert.split(" ")
    for tempToken in tempTrainAssertList:
        if tempToken not in ListOriginal:
            if tempToken in mapAndList.keys():
                if mapAndList.get(tempToken) == [] or mapAndList.get(tempToken) is None:
                    pass
                elif len(mapAndList.get(tempToken)) == 1:
                    if mapAndList.get(tempToken)[0] is None:
                        pass
                    else:
                        trainAssert = trainAssert.replace(tempToken, mapAndList.get(tempToken)[0])
                else:
                    replaceTemp = ""
                    minX = 10000000
                    llPos = mapRecordCol.get(tempToken)
                    if tempToken in mapRecordCol.keys():
                        if llPos == 0:
                            trainAssert = trainAssert.replace(tempToken, mapAndList.get(tempToken)[0])
                        else:
                            for tempA in mapAndList.get(tempToken):
                                if tempA in mapRecordCol1.keys():
                                    posT = mapRecordCol1.get(tempA)
                                    if math.fabs(llPos - posT) < minX:
                                        minX = math.fabs(llPos - posT)
                                        replaceTemp = tempA
                            if replaceTemp == "":
                                trainAssert = trainAssert.replace(tempToken, mapAndList.get(tempToken)[0])
                            else:
                                trainAssert = trainAssert.replace(tempToken, replaceTemp)
                    else:
                        trainAssert = trainAssert.replace(tempToken, mapAndList.get(tempToken)[0])
    if testAssert == trainAssert:
        return True, testAssert
    return False, trainAssert



