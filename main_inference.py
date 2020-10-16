import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import sys

#%%

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, 1)
    self.out = nn.Sigmoid()
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = pooled_output
    linear = self.linear(output)
    return self.out(linear)


model = SentimentClassifier(2)
model.load_state_dict( torch.load('./security_project/model_weights/best_model_state_binary_for_training.bin'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#%%
df = pd.read_csv("./security_project/Data/adv_reviews_clean_scores.csv")
#%%
for i in range(len(df)):
    print (df.iloc[i]['adversarial'])
#%%

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
#%%
adversarial_or_clean = 'adversarial'
scores_input = 'adversarial_scores'
#%%
scores = []
for i in range(len(df)):
    sentance = df.iloc[i][adversarial_or_clean]
    original_l = df.iloc[i]['original_label']
    encoded_review = tokenizer.encode_plus(
      sentance,
      max_length= 512,
      add_special_tokens=True,
      return_token_type_ids=False,
      padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)
    try:
        output = model(input_ids, attention_mask)
        prediction = (output>0.5).float()
        scores.append(output.item())
    except Exception as e:
        print (e)
        scores.append(original_l)
    # print (len(tokenizer.tokenize(sentance)))
    # print(f'Review text: {sentance}')
    # print(f'Sentiment  : {output.item()}')
#%%
print (scores_input, scores)
df[scores_input] = scores
#%%
print (df['adversarial_scores'].head(20))
#%%
keep = df
#%%
df.to_csv('./security_project/Data/adv_reviews_scores.csv', index=None, header=True)

#%%
df = pd.read_csv('./security_project/Data/adv_reviews_scores.csv')
#%%
print (df.columns)
#%%
print (df.head(10))
#%%
id_score = {}
for i in range(len(df)):
    id = str(df['id'].iloc[i])
    if id not in id_score:
        id_score[id] = {'clean': df['clean'].iloc[i],'adversarial': df['adversarial'].iloc[i], 'adversarial_scores' :  df['adversarial_scores'].iloc[i] , 'clean_scores' :  df['clean_scores'].iloc[i], 'original_label' :  df['original_label'].iloc[i]  }

    if id in id_score:
        if id_score[id]['original_label'] == 0:
            if id_score[id]['adversarial_scores'] <  df['adversarial_scores'].iloc[i]:
                id_score[id] = {'clean': df['clean'].iloc[i],'adversarial': df['adversarial'].iloc[i],  'adversarial_scores' :  df['adversarial_scores'].iloc[i],'clean_scores' :  df['clean_scores'].iloc[i], 'original_label' :  df['original_label'].iloc[i]  }
            else:
                pass
        else:
            if id_score[id]['adversarial_scores'] >  df['adversarial_scores'].iloc[i]:
                id_score[id] = {'clean': df['clean'].iloc[i],'adversarial': df['adversarial'].iloc[i],  'adversarial_scores' :  df['adversarial_scores'].iloc[i],'clean_scores' :  df['clean_scores'].iloc[i], 'original_label' :  df['original_label'].iloc[i]  }
            else:
                pass

#%%
for i,j in id_score.items():
    print (i,j)
    sys.exit()


#%%
df2 = pd.DataFrame.from_dict(id_score,orient='index')

df2['id'] = df2.index
df2 = df2.reset_index()
print (df2.head(2))
#%%
df2.to_csv('./security_project/Data/adv_reviews_all_scores.csv', index=None, header=True)
#%%
print (df2[['original_label', 'clean_scores', 'adversarial_scores']].head(50))
