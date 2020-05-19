### load packages

import numpy as np
import pandas as pd

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import pickle
import time
import datetime
import random

from sklearn import metrics

import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup


from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


### load device

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
    
### read review splitted df

train_X_splitted = pd.read_csv('train_X_splitted.csv')
val_X_splitted = pd.read_csv('val_X_splitted.csv')


### define tokenizer

def bert_tokenizer(df, splitted=False):
    sentences = df['cleaned_review'].tolist()
    labels = df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 200,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'      
                       )    
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    if splitted: # split reviews into multiple rows
        review_no = df['review_no'].tolist()
        review_no = torch.tensor(review_no)
        output = TensorDataset(input_ids, attention_masks, labels, review_no)
    
    else:
        output = TensorDataset(input_ids, attention_masks, labels)
    
    return output


train_bert = bert_tokenizer(train_X_splitted, splitted=True)
valid_bert = bert_tokenizer(val_X_splitted, splitted=True)


#### dataloader

batch_size = 32

train_dataloader = DataLoader(
            train_bert,
            sampler = RandomSampler(train_bert),
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            valid_bert,
            sampler = SequentialSampler(valid_bert), 
            batch_size = batch_size
        )

# test_dataloader = DataLoader(
#             test_bert,
#             sampler = SequentialSampler(test_bert),
#             batch_size = batch_size
#         )



### setup pretrained model

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)


### run on the GPU
model.to(device)


### define optimizer
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)


### initialization

epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

### helper function

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_auc_list(output, label):
    mysoftmax = nn.Softmax(dim=1)
    output_softmax = mysoftmax(output)

    _, preds = torch.max(output, dim = 1)
    
    preds = preds.cpu().numpy()
    truelabels = label.cpu().numpy()
    probas = output_softmax.cpu().numpy()
    
    return preds, truelabels, probas


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))



### start training and validation

seed_val = 2020

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

total_t0 = time.time()
training_stats = []

IS_SPLITTED = True

for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    total_train_accuracy = 0
    

    model.train()

    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        total_train_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        if IS_SPLITTED:
            # acc
            b_review_no = batch[3].to(device)
            review_no = b_review_no.to('cpu').numpy()
#             total_train_accuracy += split_accuracy(logits, label_ids, review_no)
        
        else:
            # acc
            total_train_accuracy += flat_accuracy(logits, label_ids)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader) 
#     avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    
        
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    
    if not IS_SPLITTED:
        print("  Accuracy: {0:.2f}".format(avg_train_accuracy))
        
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    preds_list = []
    truelabels_list= []
    probas_list = []
    split_stat = {} # only for splitted
    all_split_stat = pd.DataFrame(data=None, columns=['review_no', 'preds', 'label', 'probas']) # only for splitted

    for idx, batch in enumerate(validation_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        total_eval_loss += loss.item()

        logits_auc = logits.detach().cpu()
        logits = logits_auc.numpy()
        label_ids_auc = b_labels.to('cpu')
        label_ids = label_ids_auc.numpy()
        
        if IS_SPLITTED:
            b_review_no = batch[3].to(device)
            review_no = b_review_no.to('cpu').numpy()
            
#             # acc
#             total_train_accuracy += split_accuracy(logits, label_ids, review_no)


            mysoftmax = nn.Softmax(dim=1)
            output_softmax = mysoftmax(logits_auc)

            _, preds = torch.max(output_softmax, dim = 1)

            preds = preds.cpu().numpy()
            truelabels = label_ids # label.cpu().numpy()
            probas = output_softmax.cpu().numpy()[:, 1]
            
            
            pred_flat = preds.flatten() # np.argmax(preds, axis=1).flatten()
            labels_flat = label_ids.flatten()
            review_no_flat = review_no.flatten()

            for i in range(len(review_no_flat)):

                this_key = review_no_flat[i]

                if this_key not in split_stat.keys():
                    split_stat[this_key] = {
                        'preds': [preds[i]],
                        'label': labels_flat[i],
                        'probas': [probas[i]]
                    }

                else:
                    split_stat[this_key]['preds'].append(preds[i])
                    split_stat[this_key]['probas'].append(probas[i])

                    
                    
        else:
            # acc
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            
            # auc
            preds, truelabels, probas = flat_auc_list(logits_auc, label_ids_auc)
            preds_list.append(preds)
            truelabels_list.append(truelabels)
            probas_list.append(probas)
            
    
    if IS_SPLITTED:
        for k, v in split_stat.items():
            
            # if fake
            if 1 in v['preds']:
                all_split_stat = all_split_stat.append({'review_no': k, 'preds': 1, 'label': v['label'],
                                                        'probas': np.amax(v['probas'])}, ignore_index=True, )

            
            # if not fake
            else:
                all_split_stat = all_split_stat.append({'review_no': k, 'preds': 0, 'label': v['label'],
                                                        'probas': np.amin(v['probas'])}, ignore_index=True)
    
        all_split_stat['review_no'] = all_split_stat['review_no'].astype(dtype='int32')
        all_split_stat['preds'] = all_split_stat['preds'].astype(dtype='int32')
        all_split_stat['label'] = all_split_stat['label'].astype(dtype='int32')
        
        
            
    # save model
    PATH = 'models/bert_based_uncased_splitted200_model_' + str(epoch_i) + '.pth'
    torch.save(model.state_dict(), PATH)

        
#     avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    
    
    
    if IS_SPLITTED:
        probas_list = all_split_stat['probas']
        truelabels_list = all_split_stat['label']
        preds_list = all_split_stat['preds'].tolist()
    
    else:
        probas_list = np.vstack(probas_list)
        truelabels_list = np.concatenate(truelabels_list)
        preds_list = np.concatenate(preds_list)
        
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
    
    
    auc_score = metrics.roc_auc_score(truelabels_list, preds_list, average='micro')
    ap = metrics.average_precision_score(truelabels_list, preds_list, average='micro')
    
    print("  Micro AUC score: {0:.2f}".format(auc_score))
    print("  Micro AP score: {0:.2f}".format(ap))


    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    
    if IS_SPLITTED:
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. AUC': auc_score,
                'Valid. AP': ap,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'auc_stat': {
                    'probas_list': probas_list,
                    'truelabels_list': truelabels_list,
                    'preds_list': preds_list
                }
            }
        )
        
    else:
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Training Accur.': avg_train_accuracy,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Valid. AUC': auc_score,
                'Valid. AP': ap,
                'Training Time': training_time,
                'Validation Time': validation_time,
                'auc_stat': {
                    'probas_list': probas_list,
                    'truelabels_list': truelabels_list,
                    'preds_list': preds_list
                }
            }
        )


    # save results
    with open("history_bert_based_uncased_splitted200.pkl", "wb") as fout:
        pickle.dump(training_stats, fout)

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
