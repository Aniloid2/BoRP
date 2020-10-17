import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import sys

#%%



class SingleModelling():
    def __init__(self ):

        self.PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setup_classifier(self):

        class SentimentClassifier(nn.Module):
          def __init__(this, n_classes):
            super(SentimentClassifier, this).__init__()
            this.bert = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
            # this.drop = nn.Dropout(p=0.3)
            this.linear = nn.Linear(this.bert.config.hidden_size, 1)
            this.out = nn.Sigmoid()
          def forward(this, input_ids, attention_mask):
            _, pooled_output = this.bert(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            output = pooled_output
            linear = this.linear(output)
            return this.out(linear)


        model = SentimentClassifier(2)
        model.load_state_dict( torch.load('./security_project/model_weights/best_model_state_binary_for_training.bin'))


        self.model = model.to(self.device)
        self.model.eval()
        return('model setup successful')


    def do_inferance(self,sample, label):
        # single sample inferance
        sentance = sample
        original_l = label
        encoded_review = self.tokenizer.encode_plus(
          sentance,
          max_length= 512,
          add_special_tokens=True,
          return_token_type_ids=False,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )

        input_ids = encoded_review['input_ids'].to(self.device)
        attention_mask = encoded_review['attention_mask'].to(self.device)
        try:
            output = self.model(input_ids, attention_mask)
            prediction = (output>0.5).float()
            return (output.item())
        except Exception as e:
            print (e)
            return (original_l)
    def dataset_inferance(self,dataset_load = "adv_reviews_clean_scores.csv" , dataset_save = False,type='adversarial' ):
        # parse a dataset and do inferance for each sample
        main_file = pd.read_csv("./security_project/Data/"+dataset_load)
        self.scores = []
        if type not in ['adversarial','clean']:
            print ('chose clean or adversarial mate')
            sys.exit()
        else:
            for i in range(len(main_file)):
                adversarial_or_clean = type
                scores_input = adversarial_or_clean+'_scores'
                sentance = main_file.iloc[i][adversarial_or_clean]
                original_l = main_file.iloc[i]['original_label']
                output = self.do_inferance(sentance,original_l)

                self.scores.append(output)

            main_file[scores_input] = self.scores
        if dataset_save:
            main_file.to_csv('./security_project/Data/'+dataset_save, index=None, header=True)
    def best_adversarial_samples(self, dataset_load = False,dataset_save = False):

        # load a csv file and remove the adversarial samples with the lowest score for each id
        main_file = pd.read_csv('./security_project/Data/'+dataset_load)


        id_score = {}
        for i in range(len(main_file)):
            id = str(main_file['id'].iloc[i])
            if id not in id_score:
                id_score[id] = {'clean': main_file['clean'].iloc[i],'adversarial': main_file['adversarial'].iloc[i], 'adversarial_scores' :  main_file['adversarial_scores'].iloc[i] , 'clean_scores' :  main_file['clean_scores'].iloc[i], 'original_label' :  main_file['original_label'].iloc[i]  }

            if id in id_score:
                if id_score[id]['original_label'] == 0:
                    if id_score[id]['adversarial_scores'] <  main_file['adversarial_scores'].iloc[i]:
                        id_score[id] = {'clean': main_file['clean'].iloc[i],'adversarial': main_file['adversarial'].iloc[i],  'adversarial_scores' :  main_file['adversarial_scores'].iloc[i],'clean_scores' :  main_file['clean_scores'].iloc[i], 'original_label' :  main_file['original_label'].iloc[i]  }
                    else:
                        pass
                else:
                    if id_score[id]['adversarial_scores'] >  main_file['adversarial_scores'].iloc[i]:
                        id_score[id] = {'clean': main_file['clean'].iloc[i],'adversarial': main_file['adversarial'].iloc[i],  'adversarial_scores' :  main_file['adversarial_scores'].iloc[i],'clean_scores' :  main_file['clean_scores'].iloc[i], 'original_label' :  main_file['original_label'].iloc[i]  }
                    else:
                        pass


        removed_low_adv = pd.DataFrame.from_dict(id_score,orient='index')

        removed_low_adv['id'] = removed_low_adv.index
        removed_low_adv = removed_low_adv.reset_index()
        if dataset_save:
            removed_low_adv.to_csv('./security_project/Data/'+dataset_load, index=None, header=True)


#%%
