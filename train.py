import os
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from model import *
from metric import *


# ========= START Configure Environment ========= #
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 101

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Found CUDA device: {device}")

# ========= END Configure Environment   ========= #


# =================== START Read Data =================== #
print('Processing Data ...')
news_col = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entity', 'abstract_entity']
behaviors_col = ['impression_id', 'user', 'time', 'clicked_news', 'impressions']

news_df = pd.read_csv('./train/fixed_train_news.tsv', sep='\t', names=news_col, index_col='news_id')
behav_df = pd.read_csv('./train/train_behaviors.tsv', sep='\t', names=behaviors_col, index_col='impression_id')
entity_vec = pd.read_csv('./train/train_entity_embedding.vec', sep='\t', 
                         names=['WikidataId'] + list(range(100)), 
                         usecols=list(range(101)),
                         index_col='WikidataId')

# 將abstract的缺值替換成空字串
news_df['abstract'].fillna('', inplace=True)
news_df['abstract'].isna().any()

# 把click數超過200的samples移除 (2859筆佔整體資料極小部份)
safe_ids = [i for i, clicks in behav_df['clicked_news'].str.split().items() if len(clicks) <= 200]
behav_df = behav_df.loc[safe_ids]

# =================== END Read Data   =================== #


# =================== START Load Model =================== #
print('Loading pre-trained model ...')
from transformers import AutoTokenizer, AutoModel

model_name = 'roberta-base'

tokenizer = AutoTokenizer.from_pretrained(model_name)
lang_model = AutoModel.from_pretrained(model_name)
lang_model.requires_grad_(False)
lang_model.pooler.requires_grad_(True)

# =================== END Load Model   =================== #


# =================== START Build Dataset =================== #
max_text_len = 100

print("Tokenizing news ...")
news_inputs = dict()
for news_id, row in tqdm(news_df.iterrows(), total=len(news_df)):
    title, abstract = row['title'], row['abstract']
    news_inputs[news_id] = \
        tokenizer(title, abstract, 
                  padding='max_length', 
                  max_length=max_text_len, 
                  truncation=True, 
                  return_tensors='pt')


# =================== START Training =================== #
print("Building dataset ...")
from sklearn.model_selection import train_test_split

train_behav, valid_behav = train_test_split(behav_df, test_size=0.3, random_state=SEED)

train_data = TrainingDataset(news_inputs, train_behav, max_clicks=50, max_text_len=max_text_len, seed=SEED)
valid_data = TrainingDataset(news_inputs, valid_behav, max_clicks=50, max_text_len=max_text_len, seed=SEED)


print("Start training ...")
from transformers import TrainingArguments, Trainer

news_encoder = NewsEncoder(lang_model, 100)
user_encoder = UserClicksEncoder(100, 100, attn_num_heads=4)
model = RecommendationSystem(news_encoder, user_encoder, n_labels=15, max_clicks=train_data.max_clicks)
# model = model.to(device)

training_args = TrainingArguments(
    output_dir='model/',
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    lr_scheduler_type='linear',
    learning_rate=1e-3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="roc_auc",
    seed=SEED,
    data_seed=SEED
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    data_collator=NewsGatherCollator(),
    compute_metrics=compute_metric
)

trainer.train()

# =================== END Trainging   =================== #
