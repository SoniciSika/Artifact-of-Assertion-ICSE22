import numpy as np
import pandas as pd
import os
import pickle
import torch

import torch.nn as nn
import collections
from torch.utils.data import DataLoader
from data import AlignDataset
from biased_plabel import ESIM
import sys
from utils import train, validate
hidden_size = 300
dropout = 0.3
num_classes = 2
epochs = 50
batch_size = 42
lr = 0.0005
patience = 10
max_grad_norm = 10
checkpoint = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_flag = True
load_voc = False
split_num = 1
embedding_dim = 100
voc_size = 500000

def loadlabel(path):
    ret = []
    with open(path) as f:
        for line in f:
            l = line.split()
            l = [int(x) for x in l]
            ret.append(l)
    return ret

def make_voc(data, load_voc):
    if load_voc:
        f = open(os.path.join(target_dir, "word2idx"), "rb")
        word_to_indices = {}
        word_to_indices = pickle.load(f)
    else:
        query_list = ' '.join(data['query']+data['query2']).split()
        mpvoc = collections.defaultdict(int)
        for q in query_list:
            mpvoc[q] += 1
        title_list = ' '.join(data['title']).split()
        for t in title_list:
            mpvoc.setdefault(t,0)
            mpvoc[t] += 1
        word_list = []
        for k in mpvoc:
            if mpvoc[k] > 5:
                word_list.append(k)
        word_list = word_list + ['</s>', 'pos', 'pad','<unk>']
        len(word_list)
        word_set = []
        wset = set()
        for w in word_list:
            if w not in wset:
                word_set.append(w)
                wset.add(w)
        word_to_indices = {}
        indices = 0
        word_to_indices['</s>'] = 0  # end
        indices += 1 
        word_to_indices['pad'] = 1   # pad
        indices += 1 
        word_to_indices['pos'] = 2   # no find its word means not in embedmatrix or not in texts
        indices += 1 
        word_to_indices['<unk>'] = 3  # no find its word means not in embedmatrix or not in texts
        indices += 1 
        for word in word_set:
            if word in ['</s>', 'pad', 'pos','<unk>']: continue
            word_to_indices[word] = indices
            indices += 1
            if indices > voc_size: break
        print(indices, 'words in train and test texts')
        f = open(os.path.join(target_dir, "word2idx"),'wb')
        pickle.dump(word_to_indices, f)
    return word_to_indices
def data2seqs(data):
    tokens = []
    for row in data['query']:
        tokens.append(row.split(' '))

    tokens2 = []
    for row in data['query2']:
        tokens2.append(row.split(' '))
    tokens3 = []

    for i, row in enumerate(data['title']):
        tokens3.append(row.split(' '))
    sequences_query = [[word_to_indices[i] if i in word_to_indices else word_to_indices['<unk>'] for i in t ]+[0] for t in tokens]
    sequences_query2 = [[word_to_indices[i] if i in word_to_indices else word_to_indices['<unk>'] for i in t ]+[0] for t in tokens2]
    sequences_title = [[word_to_indices[i] if i in word_to_indices else word_to_indices['<unk>'] for i in t] + [0]   for t in tokens3]
    return sequences_query, sequences_query2, sequences_title



def inference(checkpoint):
    checkpoint = torch.load(checkpoint)
    valid_esmim = {
        'ids': train_idx,
        'premises': method1,
        'premises2': method2,
        'hypotheses': assertion,
        'labels': list(data_df['label']),
        'plabels':train_labels, 
        'aligns':train_align,
        'assert_test2': assert_test2_matrix,
        'test1_test2':test1_test2_matrix
    }
    Nvalid = AlignDataset(valid_esmim, padding_idx=1, max_premise_length=1050, max_hypothesis_length=1050)
    valid_loader = DataLoader(Nvalid, shuffle=False, batch_size=batch_size)
    model.load_state_dict(checkpoint["model"])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    epoch_time, valid_loss, valid_accuracy, prob, prob_p = validate(model,
                                         valid_loader,
                                         criterion)
    f =open('similarity_saved','wb')
    pickle.dump(prob_p, f)
    np.save('compatibility_saved',prob[:,0])

def train_process(checkpoint):
    
    valid_esmim = {
        'ids': train_idx[:split_num],
        'premises': method1[:split_num],
        'premises2': method2[:split_num],
        'hypotheses': assertion[:split_num],
        'labels': list(data_df['label'][:split_num]),
        'plabels':train_labels[:split_num], 
        'aligns':train_align[:split_num],
        'assert_test2': assert_test2_matrix[:split_num],
        'test1_test2':test1_test2_matrix[:split_num]
    }

    train_esmim = {
        'ids': train_idx[split_num:],
        'premises': method1[split_num:],
        'premises2': method2[split_num:],
        'hypotheses': assertion[split_num:],
        'labels': list(data_df['label'][split_num:]),
        'plabels':train_labels[split_num:], 
        'aligns':train_align[split_num:], 
        'assert_test2': assert_test2_matrix[split_num:],
        'test1_test2':test1_test2_matrix[split_num:]
    }
    Ntrain = AlignDataset(train_esmim, padding_idx=1, max_premise_length=1050, max_hypothesis_length=1050)
    Nvalid = AlignDataset(valid_esmim, padding_idx=1, max_premise_length=1050, max_hypothesis_length=1050)
    train_loader = DataLoader(Ntrain, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(Nvalid, shuffle=False, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        mode="max",
                                                        factor=0.5,
                                                        patience=0)

    best_score = 0.0
    start_epoch = 1
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]

        print("\t* Training will continue on existing model from epoch {}..."
            .format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]

    # Compute loss and accuracy before starting (or resuming) training.


    _, valid_loss, valid_accuracy, prob, prob_p = validate(model,
                                            valid_loader,
                                            criterion)
    f =open(os.path.join(target_dir, "similarity"),'wb')
    pickle.dump(prob_p, f)
    np.save(os.path.join(target_dir, "compatibility"),prob[:,0])

    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
        .format(valid_loss, (valid_accuracy*100)))

    # -------------------- Training epochs ------------------- #
    print("\n",
        20 * "=",
        "Training ESIM model on device: {}".format(device),
        20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                    train_loader,
                                                    optimizer,
                                                    criterion,
                                                    epoch,
                                                    max_grad_norm)

        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
            .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, prob, prob_p = validate(model,
                                                        valid_loader,
                                                        criterion)
        #np.argsort(prob[:,:1].squeeze())

        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n"
            .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        scheduler.step(epoch_accuracy)

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            if save_flag:
                torch.save({"epoch": epoch,
                            "model": model.state_dict(),
                            "best_score": best_score,
                            "optimizer": optimizer.state_dict(),
                            "epochs_count": epochs_count,
                            "train_losses": train_losses,
                            "valid_losses": valid_losses},
                        os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        if save_flag:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                    os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == '__main__':
    data_path = sys.argv[1]

    # data_path = '/home/sunke/data_process'
    data_pd = os.path.join(data_path, 'data')
    similarity_label = os.path.join(data_path, 'cls_labels')
    similarity_align = os.path.join(data_path, 'cls_align')
    text_similarity_matrix_assert = os.path.join(data_path, 'matrix_a_t2')
    text_similarity_matrix_method = os.path.join(data_path, 'matrix_t1_t2')
    target_dir = sys.argv[2]
    # target_dir = '/home/sunke/ESIM-pytorch-master/save_new'

    train_labels = loadlabel(similarity_label)
    train_align = loadlabel(similarity_align)
    assert_test2_matrix = pickle.load(open(text_similarity_matrix_assert,'rb'))
    test1_test2_matrix = pickle.load(open(text_similarity_matrix_method,'rb'))
    data_df = pd.read_csv(data_pd,sep='<spex>',names=['query_id','query','query_id2','query2','query_title_id','title','label'])
    data_df['query_len'] = data_df['query'].apply(lambda x: len(x.split(' ')))
    data_df['query_len2'] = data_df['query2'].apply(lambda x: len(x.split(' ')))
    data_df['title_len'] = data_df['title'].apply(lambda x: len(x.split(' ')))


    word_to_indices = make_voc(data_df, load_voc)
    print('Preparing embeddings matrix...')
    num_words = len(word_to_indices)
    embedding_matrix = np.zeros((num_words, embedding_dim))


    method1, method2, assertion = data2seqs(data_df)
    train_idx= [[data_df['query_id'][i], data_df['query_id2'][i], data_df['query_title_id'][i]] for i in range(data_df.shape[0])]


    print("\t* Loading training data...")
    embeddings = torch.tensor(embedding_matrix, dtype=torch.float).to(device)


    model = ESIM(embeddings.shape[0],
                embeddings.shape[1],
                hidden_size,
                embeddings=embeddings,
                padding_idx=1,
                dropout=dropout,
                num_classes=num_classes,
                device=device).to(device)
    mode = sys.argv[3]
    if mode == 'train':
        train_process(checkpoint)
    else:
        inference(checkpoint)