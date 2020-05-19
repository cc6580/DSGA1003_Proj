import numpy as np
import pandas as pd
import pickle

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

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

train_X = pd.read_csv('Data/train_X.csv')
train_y = pd.read_csv('Data/train_y.csv')
val_X = pd.read_csv('Data/val_X.csv')
val_y = pd.read_csv('Data/val_y.csv')

train = pd.read_csv('Data/train.csv')
val = pd.read_csv('Data/dev.csv')


def get_split(text1):
    l_total = []
    l_parcial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return l_total

train_X['cleaned_review'] = train_X['cleaned review'].apply(lambda x: x[2:-1])
train_X_sub = train_X.loc[:, ['cleaned_review']]
train_X_sub['label'] = train_y['label']
train_X_sub['review_split'] = train_X_sub['cleaned_review'].apply(get_split)
train_X_sub = train_X_sub.loc[:, ['label', 'cleaned_review', 'review_split']]


val_X['cleaned_review'] = val_X['cleaned review'].apply(lambda x: x[2:-1])
val_X_sub = val_X.loc[:, ['cleaned_review']]
val_X_sub['label'] = val_y['label']
val_X_sub['review_split'] = val_X_sub['cleaned_review'].apply(get_split)
val_X_sub = val_X_sub.loc[:, ['label', 'cleaned_review','review_split']]


import transformers

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence


def bert_tokenizer(df):
    sentences = df['cleaned_review'].tolist()
    labels = df['label'].tolist()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 128,
                            pad_to_max_length = True,
                            return_attention_mask = True,
                            return_tensors = 'pt'      
                       )    
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    output = TensorDataset(input_ids, attention_masks, labels)
    
    return output

train_bert = bert_tokenizer(train_X_sub)
valid_bert = bert_tokenizer(val_X_sub)

# dataloader
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


# setup pretrained model
from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False
)

# run on the GPU
model.to(device)

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

from transformers import get_linear_schedule_with_warmup

epochs = 4

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random

seed_val = 2020

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

training_stats = []

total_t0 = time.time()

for epoch_i in range(0, epochs):
    
#     # ========================================
#     #               Training
#     # ========================================

#     print("")
#     print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
#     print('Training...')

#     t0 = time.time()

#     total_train_loss = 0
#     total_train_accuracy = 0

#     model.train()

#     for step, batch in enumerate(train_dataloader):
#         # Progress update every 40 batches.
#         if step % 40 == 0 and not step == 0:
#             # Calculate elapsed time in minutes.
#             elapsed = format_time(time.time() - t0)
            
#             print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

#         b_input_ids = batch[0].to(device)
#         b_input_mask = batch[1].to(device)
#         b_labels = batch[2].to(device)

#         model.zero_grad()        

#         loss, logits = model(b_input_ids, 
#                              token_type_ids=None, 
#                              attention_mask=b_input_mask, 
#                              labels=b_labels)

#         total_train_loss += loss.item()
        
#         logits = logits.detach().cpu().numpy()
#         label_ids = b_labels.to('cpu').numpy()
#         total_train_accuracy += flat_accuracy(logits, label_ids)

#         loss.backward()

#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

#         optimizer.step()

#         scheduler.step()

#     avg_train_loss = total_train_loss / len(train_dataloader) 
#     avg_train_accuracy = total_train_accuracy / len(train_dataloader)
    
    
#     training_time = format_time(time.time() - t0)

#     print("")
#     print("  Average training loss: {0:.2f}".format(avg_train_loss))
#     print("  Accuracy: {0:.2f}".format(avg_train_accuracy))
#     print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()
    
    LOAD_MODEL = True
    MODEL_PATH = 'models/bert_based_uncased_first128_model_'
    
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH + str(epoch_i) + '.pth'))
    
    model.to(device)

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        with torch.no_grad():        
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
	
#     # save model
#     PATH = 'models/bert_based_uncased_first128_model_' + str(epoch_i) + '.pth'
#     torch.save(model.state_dict(), PATH)

        
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Training Accur.': avg_train_accuracy,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )


    # save results
    with open("history_bert_based_uncased_first128_auc.pkl", "wb") as fout:
        pickle.dump(training_stats, fout)

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


