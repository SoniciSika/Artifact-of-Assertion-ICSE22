"""
Utility functions for training and validating models.
"""
import numpy as np
import time
import torch

import torch.nn as nn

from tqdm import tqdm
def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range =\
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths))).to(sequences_lengths.device)
    
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences.

    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).

    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)

def biased_sum(tensor, weights, bias, mask):
    weighted_sum = weights.bmm(tensor)
    biased_sum = bias.bmm(tensor)
    
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask
# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def correct_predictions(output_probabilities, targets,pointer=False):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    if pointer:
        al = 1
        correct = 0
        wrong = 0
        _, out_classes = output_probabilities.max(dim=-1)
        a = output_probabilities.view(-1,output_probabilities.size()[-1])
        b = targets.reshape(-1)
        for i, x in enumerate(b):
            if x.item() != -1:
                al += 1
                if a[i].argmax().item() == x.item():
                    correct += 1
                else:

                    wrong += 1
        print(correct/al)
        return output_probabilities.cpu().numpy().tolist()
        #correct = (out_classes == targets[:,:out_classes.size(-1)]).sum()
    else:
        _, out_classes = output_probabilities.max(dim=1)
        correct = (out_classes == targets).sum()
    return correct.item()
def padding_sequence(sequence, max_len):
    return sequence + [0] * (max_len - len(sequence))
def train(model,
          dataloader,
          optimizer,
          criterion,
          epoch_number,
          max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        criterion: A loss criterion to use for training.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        premises = batch["premise"].to(device)
        premises_lengths = batch["premise_length"].to(device)
        premises2 = batch["premise2"].to(device)
        premises_lengths2 = batch["premise_length2"].to(device)
        
        hypotheses = batch["hypothesis"].to(device)
        hypotheses_lengths = batch["hypothesis_length"].to(device)
        labels = batch["label"].to(device)


        assert_test2_matrix = batch["assert_test2_matrix"].to(device)
        test1_test2_matrix = batch["test1_test2_matrix"].to(device)
        
        optimizer.zero_grad()
        if batch.get("plabel",None) is not None:
            plabel = batch["plabel"].to(device)
        else:
            plabel = None
        if batch.get("aligns",None) is not None:
            aligns = batch["aligns"].to(device)
        else:
            aligns = None #bsz, len
        
        if plabel is not None:
            plogits, pplogits , logits, probs= model(premises,
                            premises_lengths,
                            premises2,
                            premises_lengths2,
                            
                            hypotheses,
                            hypotheses_lengths,
                            assert_test2_matrix,
                            test1_test2_matrix)
            #for i in range
        
            pp = torch.zeros_like(plogits)
            for i in range(pp.size(0)):
                for j in range(pp.size(1)):
                    if aligns[i][j].item() != -1:
                        pp[i][j] = pplogits[i][aligns[i][j]]

            
            plogits_ = plogits + pp
            plogits_ /= 2
            pprob = torch.log(plogits_+1e-20)
            loss = criterion(pprob.view(-1,plogits.size()[-1]), plabel[:,:torch.max(hypotheses_lengths)].reshape(-1))
            plabels = plabel[:,:torch.max(hypotheses_lengths)]
            loss += criterion(logits, labels)
        else:
            try:
                logits, probs = model(premises,
                                    premises_lengths,
                                    premises2,
                                    premises_lengths2,
                                    hypotheses,
                                    hypotheses_lengths)
                
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception
            
            loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        if plabel is not None:
            correct_preds += correct_predictions(probs, labels)
            #correct_predictions(plogits, plabels,True)
        else:
            correct_preds += correct_predictions(probs, labels)

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1),
                              running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    
    pred_l = []
    pred_p = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            premises2 = batch["premise2"].to(device)
            premises_lengths2 = batch["premise_length2"].to(device)
            
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)
            assert_test2_matrix = batch["assert_test2_matrix"].to(device)
            test1_test2_matrix = batch["test1_test2_matrix"].to(device)
            if batch.get("plabel",None) is not None:
                plabel = batch["plabel"].to(device)
            else:
                plabel = None
            if batch.get("aligns",None) is not None:
                aligns = batch["aligns"].to(device)
            else:
                aligns = None #bsz, len
            
            if plabel is not None:
                plogits, pplogits, logits, probs = model(premises,
                                premises_lengths,
                                premises2,
                                premises_lengths2,
                                
                                hypotheses,
                                hypotheses_lengths,
                                assert_test2_matrix,
                                test1_test2_matrix)
                aligns = aligns[:,:plogits.size(1)]
                pp = torch.zeros_like(plogits)
                for i in range(pp.size(0)):
                    for j in range(pp.size(1)):
                        if aligns[i][j].item() != -1:
                            pp[i][j] = pplogits[i][aligns[i][j]]
                plogits += pp
                plogits /= 2
                plogits = torch.log(plogits+1e-20)
                loss = criterion(plogits.view(-1,plogits.size()[-1]), plabel[:,:torch.max(hypotheses_lengths)].reshape(-1))
                plabels = plabel[:,:torch.max(hypotheses_lengths)]
                # loss += criterion(logits, labels)
                #prob = nn.functional.softmax(plogits, dim=-1)
                #_, out_classes = plogits.max(dim=-1)
                pred_l.append(np.array(probs.cpu()))
                #pred_l.append(np.array([0]))
                #pred_p.append(np.array(prob[:,:,0].cpu()))
            else:
                logits, probs = model(premises,
                                    premises_lengths,
                                    premises2,
                                    premises_lengths2,
                                    
                                    hypotheses,
                                    hypotheses_lengths)
                
                loss = criterion(logits, labels)
            
                pred_l.append(np.array(probs.cpu()))
            
            running_loss += loss.item()
            if plabel is not None:
                running_accuracy += correct_predictions(probs, labels)
                pred_p.append(correct_predictions(plogits, plabels,True))
            else:
                running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy, np.vstack(pred_l), pred_p


def predict(model, dataloader, criterion):
    # Switch to evaluate mode.
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    
    pred_l = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premise"].to(device)
            premises_lengths = batch["premise_length"].to(device)
            hypotheses = batch["hypothesis"].to(device)
            hypotheses_lengths = batch["hypothesis_length"].to(device)
            labels = batch["label"].to(device)

            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)
            
            pred_l.append(np.array(probs.cpu()))
            
            running_loss += loss.item()
            running_accuracy += correct_predictions(probs, labels)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    return epoch_time, epoch_loss, epoch_accuracy, np.vstack(pred_l)