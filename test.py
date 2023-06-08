import os
from datetime import date, datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import TrainingArguments, Trainer

from dataset import *
from model import *
from metric import *


# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 101
CKPT_PATH = "model/checkpoint-98856"
ENCODING_DIM = 100
MAX_CLICKS = 50  # It's better to be identical with the training setting
N_LABELS = 15

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Found CUDA device: {device}")


print('Processing Data ...')
news_col = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entity', 'abstract_entity']
behaviors_col = ['impression_id', 'user', 'time', 'clicked_news', 'impressions']

news_df = pd.read_csv('./test/fixed_test_news.tsv', sep='\t', names=news_col, index_col='news_id')
behav_df = pd.read_csv('./test/test_behaviors.tsv', sep='\t', names=behaviors_col, index_col='impression_id')
entity_vec = pd.read_csv('./test/test_entity_embedding.vec', sep='\t', 
                         names=['WikidataId'] + list(range(100)), 
                         usecols=list(range(101)),
                         index_col='WikidataId')

# 將abstract的缺值替換成空字串
news_df['abstract'].fillna('', inplace=True)
news_df['abstract'].isna().any()


print('Loading language model ...')
from transformers import AutoTokenizer, AutoConfig, AutoModel

lang_model_name = 'roberta-base'

tokenizer = AutoTokenizer.from_pretrained(lang_model_name)
lang_model_config = AutoConfig.from_pretrained(lang_model_name)
lang_model = AutoModel.from_config(lang_model_config)


print("Tokenizing news ...")
max_text_len = 100
news_inputs = dict()
for news_id, row in tqdm(news_df.iterrows(), total=len(news_df)):
    title, abstract = row['title'], row['abstract']
    news_inputs[news_id] = \
        tokenizer(title, abstract, 
                  padding='max_length', 
                  max_length=max_text_len, 
                  truncation=True, 
                  return_tensors='pt')
    

print("Building dataset ...")
test_data = TestingDataset(news_inputs, behav_df, max_clicks=MAX_CLICKS, max_text_len=max_text_len, seed=SEED)


print(f"Loading trained model from \"{CKPT_PATH}\" ...")
news_encoder = NewsEncoder(lang_model, ENCODING_DIM)
user_encoder = UserClicksEncoder(ENCODING_DIM, ENCODING_DIM, attn_num_heads=4)
model = RecommendationSystem(news_encoder, user_encoder, n_labels=N_LABELS, max_clicks=MAX_CLICKS)

model_path = os.path.join(CKPT_PATH, "pytorch_model.bin")
model.load_state_dict(torch.load(model_path))

training_args = TrainingArguments(
    output_dir='model/',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    seed=SEED,
    data_seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=NewsGatherCollator(),
    compute_metrics=compute_metric
)


print("Predicting the test data ...")
test_preds = trainer.predict(test_data)

print(test_preds.predictions.shape)

now = datetime.strftime(datetime.now(), '%m%d_%H%M%S')
submission = pd.DataFrame(data=test_preds.predictions, columns=[f'p{i}' for i in range(1, 16)])
submission.to_csv(f"submission/output-{now}.csv", index_label='index')
