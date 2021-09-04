from tqdm import tqdm
import regex as re
import string
import numpy as np
import pickle
import os
PUNKS = set(a for a in string.punctuation)
pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+'
def punt_lower(text):
    result_list = re.split(pattern, text)
    return [r.lower() for r in result_list]
def tokenlize_code_node(text):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
    ret = []
    for m in matches:
        ret.extend(punt_lower(m.group(0)))
    return ret
def canbefind(text, word):
    for x in text.split():
        if x == word:
            return True
    return False

def subworded(tok1 , tok2):
    lazy_word = ['assert']
    if tok2.lower().find(tok1.lower()) != -1 or tok1.lower().find(tok2.lower()) != -1:
        return True
    tok1 = ''.join(filter(lambda x: x.isalpha(), tok1))
    tok2 = ''.join(filter(lambda x: x.isalpha(), tok2))
    
    tokenlized_tok1 = tokenlize_code_node(tok1)
    tokenlized_tok2 = tokenlize_code_node(tok2)
    for t in tokenlized_tok1:
        if t in tokenlized_tok2 and t not in lazy_word:
            return True
    return False

def read_data(data_config, retrieval_output):
    with open(data_config) as f:
        paths = f.read().split('\n')
    train_method = paths[0]
    test_method = paths[1]
    train_assert = paths[2]
    test_assert = paths[3]
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

    with open(os.path.join(retrieval_output), "retrieval_train_saved") as f:
        saved_train = f.read().split('\n')
    with open(os.path.join(retrieval_output), "retrieval_test_saved") as f:
        saved_test = f.read().split('\n')
    
    return contentTrainMethod, contentTestMethod, contentTrainAssert, contentTestAssert, saved_train, saved_test
def make_data(test_method, retrieval_method, test_assert, retrieval_assert, saved):
    with open(os.path.join(output_path, 'data'),'w+') as f2, open(os.path.join(output_path,'cls_labels'),'w+') as f4, open(os.path.join(output_path, 'cls_align'),'w+') as f5, open(os.path.join(output_path, 'matrix_a_t2'),'wb') as f6, open(os.path.join(output_path, 'matrix_t1_t2'),'wb') as f7:
        subword_matrix_a_tokmethod2s = []
        subword_matrix_tokmethod1_tokmethod2s = []
        cnt = 0
        for i in tqdm(range(0,len(test_method))):
            s = saved[i].strip().split()
            tokmethod2 = test_method[i]
            tokmethod1 = retrieval_method[int(s[1])]
            a = retrieval_assert[int(s[1])].split()
            b = test_assert[i].split()
            subword_matrix_tokmethod1_tokmethod2 = []
            subword_matrix_a_tokmethod2 = []
            flag = 0
            if len(a)==len(b):
                if ' '.join(a) == ' '.join(b):
                    f2.writelines(str(i)+'<spex>'+tokmethod2+'<spex>'+str(i)+'<spex>'+tokmethod1+'<spex>'+str(i)+'<spex>'+' '.join(a) +'<spex>'+str(1)+'\n')
                    for ii in range(len(a)):
                        f4.write('-1 ')
                        f5.write('-1 ')
                    f5.write('\n')
                    f4.write('\n')
                else:
                    lost = 0
                    fd = 0
                    na = []
                    nb = []
                    for ii in range(len(a)):
                        if canbefind(tokmethod1, a[ii]) and not canbefind(tokmethod2, a[ii]) and canbefind(tokmethod2, b[ii]):
                            # print(b[ii])
                            cnt += 1
                            flag = 1
                            na.append(b[ii])
                        else:
                            na.append(a[ii])
                    
                    for ii in range(len(a)):
                        if not canbefind(tokmethod2, a[ii]) and canbefind(tokmethod2, b[ii]):
                            for j, tok in enumerate(tokmethod2.split()):
                                if tok == b[ii]:
                                    f4.write(str(j)+' ')
                                    break
                            fg = 0
                            for j, tok in enumerate(tokmethod1.split()):
                                if tok == a[ii]:
                                    fg = 1
                                    f5.write(str(j)+' ')
                                    break
                            if fg == 0:
                                f5.write('-1 ')
                            
                        else:
                            f4.write('-1 ')
                            f5.write('-1 ')
                            
                    f4.write('\n')
                    f5.write('\n')

                    if ' '.join(na) == ' '.join(b):
                        f2.writelines(str(i)+'<spex>'+tokmethod2+'<spex>'+str(i)+'<spex>'+tokmethod1+'<spex>'+str(i)+'<spex>'+' '.join(a)+'<spex>'+str(1)+'\n')
                        
                        
                    else:
                        f2.writelines(str(i)+'<spex>'+tokmethod2+'<spex>'+str(i)+'<spex>'+tokmethod1+'<spex>'+str(i)+'<spex>'+' '.join(a)+'<spex>'+str(0)+'\n')
                        # for ii in range(len(a)):
                        #     f5.write('-1 ')
                        #     f4.write('-1 ')
                            
                        # f5.write('\n')
                        # f4.write('\n')
                    
            else:
                f2.writelines(str(i)+'<spex>'+tokmethod2+'<spex>'+str(i)+'<spex>'+tokmethod1+'<spex>'+str(i)+'<spex>'+' '.join(a)+'<spex>'+str(0)+'\n')
                for ii in range(len(a)):
                    f5.write('-1 ')
                    f4.write('-1 ')
                    
                f5.write('\n')
                f4.write('\n')
            if flag == 1:
                for ii in range(len(a)):
                    tmp = []
                    for j, tok in enumerate(tokmethod2.split()):
                        if subworded(tok, a[ii]):
                            tmp.append(j)
                    subword_matrix_a_tokmethod2.append(tmp)    
                method1split = tokmethod1.split()
                for ii in range(len(method1split)):
                    tmp = []
                    for j, tok in enumerate(tokmethod2.split()):
                        if subworded(tok, method1split[ii]):
                            tmp.append(j)
                    subword_matrix_tokmethod1_tokmethod2.append(tmp)
            
            subword_matrix_a_tokmethod2s.append(subword_matrix_a_tokmethod2)
            subword_matrix_tokmethod1_tokmethod2s.append(subword_matrix_tokmethod1_tokmethod2)

        pickle.dump(subword_matrix_a_tokmethod2s, f6)
        pickle.dump(subword_matrix_tokmethod1_tokmethod2s, f7)
    
if __name__ == '__main__':
    import sys
    input_config = sys.argv[1]
    retrieval_output = sys.argv[2]
    contentTrainMethod, contentTestMethod, contentTrainAssert, contentTestAssert, saved_train, saved_test = read_data(input_config, retrieval_output)
    
    output_path = sys.argv[3]
    mode = sys.argv[4]
    if mode == 'train':
        make_data(contentTrainMethod, contentTrainMethod, contentTrainAssert, contentTrainAssert, saved_train)
    else:
        make_data(contentTestMethod, contentTrainMethod, contentTestAssert, contentTrainAssert, saved_test)
        