#%%

import security_project.bert_classification.main_inference as Setup

#%%

# import the class from the y file
Model = Setup.SingleModelling()
# send the model to memory and load tokenizer etc
Model.setup_classifier()
# simple inferance example, returns a single score float between 0 and 1
print ('prediction output',Model.do_inferance( sample = 'hi i love you', label = 0))
# dataset_inferance takes either the type = clean column or the type = adversarial column and parses each sample present in the dataset_load = "adv_reviews_clean_scores.csv"
# through the model that has previeously been setup in gpu memory with setup_classifier, in dataset_inferance each sample is parsed to the do_inferance function
Model.dataset_inferance(dataset_load = "adv_reviews_clean_scores.csv" , type='clean') # dataset_save = "adv_reviews_clean_scores.csv" (save to itself)
# after doing inferance on the clean samples do inferance on all the adversarial samples
Model.dataset_inferance(dataset_load = "adv_reviews_clean_scores.csv" , type='adversarial') # dataset_save = "adv_reviews_clean_scores.csv" (save to itself)
# at this point we have both the adversarial_scores and clean_scores column. we want to chose for each id the best adversarial sample, the best_adversarial_samples dose this
# the outputted csv dataset file (dataset_save = "adv_reviews_scores.csv") from dataset_inferance is used as input.
Model.best_adversarial_samples(dataset_load =  "adv_reviews_clean_scores.csv" ) # dataset_save = "adv_reviews_scores.csv" (save to separate file)

#%% # checkout the files!

main_file = pd.read_csv("./security_project/Data/adv_reviews_clean_scores.csv")
print (main_file.head(5))

main_file = pd.read_csv("./security_project/Data/adv_reviews_scores.csv")
print (main_file.head(5))
