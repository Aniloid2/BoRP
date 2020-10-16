# %% codecell
import pandas as pd
import sys
import transformers
import seaborn as sns
# %% codecell
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
# all_scores = pd.read_csv('adv_reviews_all_scores.csv')
all_scores = pd.read_csv('./security_project/Data/adv_reviews_all_scores.csv')
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#%%
# calculate bit flips
count_bit_flips = 0
for i in range(len(all_scores)):
    if round(all_scores['clean_scores'].iloc[i]) == round(all_scores['adversarial_scores'].iloc[i]):
        # print ('no bit flip',all_scores['clean_scores'].iloc[i],all_scores['adversarial_scores'].iloc[i])
        pass
    else:
        # print ('bit flip',all_scores['clean_scores'].iloc[i],all_scores['adversarial_scores'].iloc[i])
        new_df.concat(all_scores.iloc[i])
        count_bit_flips +=1

print (count_bit_flips,len(all_scores) )
print (len(new_df))

#%% keep only correct samples
condition =  round(all_scores['clean_scores']) == round(all_scores['original_label'])
correct = all_scores[condition]
print (len(correct))

#%% # keep only samples that cause the bit flip
condition2 = round(correct['clean_scores'])!= round(correct['adversarial_scores'])
bitflip_samples_correct = bitflip_samples[condition2]
print (len(bitflip_samples_correct))

#%% calculate the class balance/imbalance
print (bitflip_samples_correct.tail(50))
# it has 1994 samples, 1936 as 1 and 58 as 0 it's a strong class imbalance
#%%
condition3 = bitflip_samples_correct['original_label'] == 1
positives = bitflip_samples_correct[condition3]
print (len(positives))

#%%
bitflip_samples_correct.to_csv('./security_project/Data/adv_bitflip_samples.csv', index=None, header=True)

#%%

token_lens_clean = []
token_lens_adv = []
for txt in all_scores.clean:
  tokens = tokenizer.encode(txt, max_length=512,truncation=True)
  token_lens_clean.append(len(tokens))

for txt in all_scores.adversarial:
  tokens = tokenizer.encode(txt, max_length=512,truncation=True)
  token_lens_adv.append(len(tokens))
#%%
sns.histplot(token_lens_clean,kde=True)
plt.title('Clean samples token distribution')
plt.xlim([0, 512]);
plt.ylabel('Number of samples');
plt.xlabel('Token count');
plt.savefig('./security_project/Results/Analysis/Clean_samples_distribution.png')
#%%
sns.histplot(token_lens_adv,kde=True, color ="#FFDD00" )
plt.title('Adversarial samples token distribution')
plt.xlim([0, 512]);
plt.ylabel('Number of samples');
plt.xlabel('Token count');
plt.savefig('./security_project/Results/Analysis/Adversarial_samples_distribution.png')


# %% codecell
all_scores['clean_correct'] = (round(all_scores['clean_scores']) == all_scores['original_label'])
all_scores['adversarial_correct'] = (round(all_scores['adversarial_scores']) == all_scores['original_label'])
all_scores.head()
# %% codecell
print('Clean Data Accuracy: ' + str((all_scores['clean_correct'].value_counts(normalize=True) * 100)[True]))
print('Adversarial Data Accuracy: ' + str((all_scores['adversarial_correct'].value_counts(normalize=True) * 100)[True]))
# %% codecell
