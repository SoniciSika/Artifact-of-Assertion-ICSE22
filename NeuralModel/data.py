import torch
from torch.utils.data import Dataset
import numpy as np
def padding_sequence(sequence, max_len):
    return sequence + [0] * (max_len - len(sequence))
def padding_matrix(matrix, max_len1, max_len2):

    new_matrix = np.zeros([max_len1, max_len2])

    for i in range(min(len(matrix), 1050)):
        for j in matrix[i]:
            if j < 1050:
                new_matrix[i][j] = 1
                break
    return new_matrix
class AlignDataset(Dataset):
    def refine_labels(self,data):
        new_labels = []
        for l in data["plabels"]:
            nl = []
            for label in l:
                if label >=1045:
                    nl.append(-1)
                else:
                    nl.append(label) 
            new_labels.append(nl)
        data["plabels"] = new_labels
        return data
    def refine(self, data):
        new_data = {
            'ids': [],
            'premises': [],
            'premises2': [],
            'hypotheses': [],
            'labels': [],
            'plabels':[], 
            'aligns': [],
            'assert_test2': [],
            'test1_test2':[]
        }

        for i, plabels in enumerate(data["plabels"]):
            if max(plabels) == -1:
                continue
            for k in data:
                new_data[k].append(data[k][i])
        return new_data

    def clean_long(self, data):
        new_data = {
            'ids': [],
            'premises': [],
            'premises2': [],
            'hypotheses': [],
            'labels': [],
            'plabels':[], 
            'aligns': [],
            'assert_test2': [],
            'test1_test2':[]
        }
        for i in range(len(data["plabels"])):
            if len(data['premises'][i]) >= 1050:
                continue
            if len(data['premises2'][i]) >= 1050:
                continue
            if len(data['hypotheses'][i]) >= 1050:
                continue
            for k in data:
                new_data[k].append(data[k][i])
        return new_data

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        data = self.clean_long(data)
        self.premises_lengths = [min(len(seq), max_premise_length) for seq in data["premises"]]
        self.premises_lengths2 = [min(len(seq), max_premise_length) for seq in data["premises2"]]
        
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(max(self.premises_lengths), max(self.premises_lengths2))
        
        self.hypotheses_lengths = [min(len(seq), max_hypothesis_length) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)
        self.num_sequences = len(data["premises"])
        

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "premises2": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                    
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                    
                     "plabels": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * -1,
                    
                     "aligns": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * -1,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}
                     
        self.assert_test2_martix = data["assert_test2"]
        self.test1_test2_martix = data["test1_test2"]
        
        for i, premise in enumerate(data["premises"]):
            hypothesis = data["hypotheses"][i]
            plabels = data['plabels'][i] +[-1] # pl us 1 for </s>
            aligns = data['aligns'][i] +[-1] # plus 1 for </s>
            
            if len(plabels) != len(hypothesis):
                continue
            if len(aligns) != len(hypothesis):
                continue
            
            
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])
            premise2 = data["premises2"][i]
            
            end = min(len(premise2), self.max_premise_length)
            self.data["premises2"][i][:end] = torch.tensor(premise2[:end])
            
            
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])
            
            
            self.data["plabels"][i][:end] = torch.tensor(plabels[:end])
            self.data["aligns"][i][:end] = torch.tensor(aligns[:end])
            
        self.final_len = len(self.data["ids"])
    def __len__(self):
        return self.final_len

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "premise2": self.data["premises2"][index],
                "premise_length2": min(self.premises_lengths2[index],
                                      self.max_premise_length),
                
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index],
                "plabel":self.data["plabels"][index],
                "aligns":self.data["aligns"][index],
                "assert_test2_matrix": padding_matrix(self.assert_test2_martix[index], self.max_hypothesis_length, self.max_premise_length),
                "test1_test2_matrix": padding_matrix(self.test1_test2_martix[index], self.max_hypothesis_length, self.max_premise_length)
                
                }