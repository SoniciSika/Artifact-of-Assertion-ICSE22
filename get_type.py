import javalang
import javalang.tree
cnt = 0
ListKeyWordsVarAndMethod=["assertEquals","org.junit.Assert","assertTrue","assertFalse",
"assertNotNull","assertThat","assertNull","assertArrayEquals","assertSame"]
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
def FindTe(t):
	c = 0
	for i,x in enumerate(t):
		if x == '{':
			c+= 1
		elif x =='}':
			c -= 1
			if c == 0:
				return i	
def get_correct_format_and_focal_name(method, mode):
    if mode == 0:
        content = method.strip()
        tempL=content.replace('\"<AssertPlaceHolder>\" ',"True")

        if len(tempL.split("\"<FocalMethod>\"")) > 1:
            if len(tempL.split("\"<FocalMethod>\"")[1].split()) > 1:
                focal_name = tempL.split("\"<FocalMethod>\"")[1].split()[1]
            else:
                focal_name = 'palce_holder<>'
        else:
            focal_name = 'palce_holder<>'
        tempL=tempL.split("\"<FocalMethod>\"")[0]
        
        contentC = "public class Main{" + tempL + "}"
        return contentC, focal_name
    else:
        content = method.strip()
        te = FindTe(method)
        if te == None:
            return "", ""
        tempL=content[:te+1].replace('\"<AssertPlaceHolder>\" ',"True")
        if len(content[te+1:].split()) > 1:
            focal_name = content[te+1:].strip().split()[0]
        else:
            focal_name = 'palce_holder<>'
        contentC = "public class Main{" + tempL + "}"
        return contentC, focal_name

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
            #print("aaa")
            # print(i+1)
            # print(len(content))
            if i+1!=len(content):
                return i+1
            else:
                return 0
    # print(stack.size())
    # print(stack.items)
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
        #print(1,topend)
        newList = getNewList(contentList, 0, topend + 1)
        #print(2,newList)
        newList.extend(contentList[topend + 1:])
        contentList = newList
        #print(contentList)
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
def newGetComplain(content):
    pos=getFuncStart(content)
    outString=getContentComplain(content,pos)
    return outString

def getListType(method, mode=1):
    global cnt, ListKeyWordsVarAndMethod
    
    # content=newGetComplain(method)
    # contentC = "public class Main{" + content + "}"
    contentC, focal_name = get_correct_format_and_focal_name(method, mode)
    
    try:
        tree = javalang.parse.parse(contentC)
    except Exception:
        cnt += 1
        return ['???'], ['???'], ['???']
    methodName = method.strip().split()[0]
    ListMethod = []
    ListVariable = []
    ListLiteral = []
    for path, node in tree:
        children = node
        if isinstance(children, javalang.tree.MethodInvocation):
            if children.member not in ListKeyWordsVarAndMethod:
                ListMethod.append(children.member)
            if children.qualifier not in ListKeyWordsVarAndMethod:
                ListVariable.append(children.qualifier)
        if isinstance(children, javalang.tree.ConstructorDeclaration):
            if methodName!=children.name:
                ListMethod.append(children.name)
        if isinstance(children, javalang.tree.Literal):
            if children.value != '"<AssertPlaceHolder>"':
                ListLiteral.append(children.value)
        if isinstance(children, javalang.tree.MemberReference):
            ListVariable.append(children.qualifier)
            ListVariable.append(children.member)
        if isinstance(children, javalang.tree.VariableDeclarator):
            ListVariable.append(children.name)
    RListMethod=[]
    for l in ListMethod:
        if l not in RListMethod and l!="":
            RListMethod.append(l)
    RListVariable = []
    for l in ListVariable:
        if l not in RListVariable and l != "":
            RListVariable.append(l)
    RListLiteral = []
    for l in ListLiteral:
        if l not in RListLiteral:
            RListLiteral.append(l)
    RListMethod.append(focal_name)
    return RListMethod,RListVariable,RListLiteral
# with open('')