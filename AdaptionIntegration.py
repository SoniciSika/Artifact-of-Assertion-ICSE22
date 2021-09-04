import numpy as np
from tqdm import tqdm
import pickle
import sys
from get_type import getListType
from os.path import join
def read_data(input_config, retrieval_result_path, neural_result_path):
    with open(input_config) as f:
        paths = f.read().split('\n')
    train_method = paths[0]
    test_method = paths[1]
    train_assert = paths[2]
    test_assert = paths[3]

    retrieval_index = os.path.join(retrieval_result_path, "retrieval_train_saved")

    similarity_saved = os.path.join(retrieval_result_path, "similarity_saved")
    compatibility_saved = os.path.join(retrieval_result_path, "compatibility_saved")


    fTrainMethod=open(train_method,'r',encoding="utf-8")
    fTestMethod=open(test_method,'r',encoding="utf-8")
    contentTrainMethod=fTrainMethod.read().rstrip("\n").split("\n")
    contentTestMethod=fTestMethod.read().rstrip("\n").split("\n")
    fTrainMethod.close()
    fTestMethod.close()

    fTrainAssert=open(train_assert,'r',encoding="utf-8")
    fTestAssert=open(test_assert,'r',encoding="utf-8")
    contentTrainAssert=fTrainAssert.read().rstrip("\n").split("\n")
    contentTestAssert=fTestAssert.read().rstrip("\n").split("\n")
    fTrainAssert.close()
    fTestAssert.close()

    saved = open(retrieval_index).readlines()
    saved = [x.split(' ') for x in saved]

    f = open(similarity_saved,'rb')
    pscores = pickle.load(f)
    scores = np.load(compatibility_saved)

    return contentTrainMethod, contentTestMethod, contentTrainAssert, contentTestAssert, saved, pscores, scores

threshold = 0.95

def canbefind(text, word):
    for x in text.split():
        if x == word:
            return True
    return False

def inlist(token):
    onelist = ['junit','Assert','instanceof','==','!=','<=','>=','org',',','.','(',')']
    if token.find('assert') != -1:
        return True
    if token in onelist:
        return True
    return False
def type_of_token(token, methodInvo, Variable, Literal):
    if token in methodInvo:
        return 0
    elif token in Variable:
        return 1
    elif token in Literal:
        return 2
    return 3
def eq(a, b, tokmethod1, tokmethod2, i, mode):

    if mode == 0:
        if ' '.join(a) ==  ' '.join(b):
            retrieve.write(' '.join(a) + '\n')
            return 3
        else:
            retrieve.write(' '.join(a) + '\n')
            return 2
    if mode == 0:
        retrieve.write(' '.join(a) + '\n')
        return 2
    global p,n, pn1, pn2,onelist
    if len(a) != len(b):
        
        retrieve.write(' '.join(a) + '\n')
        return 1
    # if ' '.join(a) == ' '.join(b):
    #     return 0
    
    na = []
    adaption = 0
    for ii in range(len(a)):
        if not canbefind(tokmethod1, a[ii]) and canbefind(tokmethod2, a[ii]) and not inlist(a[ii]): # 
            methodInvo, Variable, Literal = getListType(tokmethod2, 0)
            type_origin = type_of_token(a[ii], methodInvo, Variable, Literal)
            s = pscores[int(i/42)][i%42][ii]
            index_s = sorted(range(len(s)), key=lambda k: s[k], reverse=True)[:1]

            correct_index = []
            
            methodInvo, Variable, Literal = getListType(tokmethod1, 0)
            for index in index_s:
                if index < len(tokmethod1.split()) and type_of_token(tokmethod1.split()[index], methodInvo, Variable, Literal) == type_origin:
                    correct_index.append(index)
            flag = 0
            for index in correct_index:
                # 
                if index < len(tokmethod1.split()) and not inlist(tokmethod1.split()[index]):
                    na.append(tokmethod1.split()[index])
                    flag = 1
                    adaption = 1
                    break
            if flag == 0:
                na.append(a[ii])
        else:
            na.append(a[ii])
    if mode == 1 and adaption == 1:
        adaption_index.write(str(i)+'\n')
    retrieve.write(' '.join(na) + '\n')
    if ' '.join(na) ==  ' '.join(b):

        return 3
    else:
        
        return 2


def R(s, e, mode):
    for i in range(s,e):
        bestidx = int(saved[i][1])
        b = contentTestAssert[i].strip().split()
        a = contentTrainAssert[bestidx].strip().split()
        tokmethod1 = contentTrainMethod[bestidx]
        tokmethod2 = contentTestMethod[i]
        if mode <= 1:
            judge = True
        else:
            judge = scores[i] <= threshold
        if  judge:
            eq(a,b,tokmethod2,tokmethod1, i, mode)
        else:
            learning_index.write(str(i)+'\n')
            retrieve.write('\n')
    retrieve.close()
if __name__ == '__main__':

    input_config = sys.argv[1]
    retrieval_result_path = sys.argv[2]
    neural_result_path = sys.argv[3]
    threshold = float(sys.argv[4])
    adaption_integration_nn_result = retrieval_result_path
    contentTrainMethod, contentTestMethod, contentTrainAssert, contentTestAssert, saved, pscores, scores = read_data(input_config, retrieval_result_path, neural_result_path)
    
    s =  0
    e =  len(contentTestAssert)
    
    # retrieve = open('retrieval.txt', 'w+')
    # R(s,e, mode=0)
    
    adaption_index = open(join(adaption_integration_nn_result, 'adaption_index.txt'), 'w+')
    retrieve = open(join(adaption_integration_nn_result,'RAadapt-NN.txt'), 'w+')
    R(s,e, mode=1)
    adaption_index.close()

    learning_index = open(join(adaption_integration_nn_result, 'CombineLearningIndex.txt'), 'w+')
    retrieve = open(join(adaption_integration_nn_result,'Combine.txt'), 'w+')
    R(s,e, mode=2)
    learning_index.close()