import random
from typing import Any, List, Dict, Callable
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import default_collate
from transformers.data.data_collator import DataCollatorMixin

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 news: pd.DataFrame, 
                 behaviors: pd.DataFrame, 
                 tokenizer: Callable, 
                 max_clicks: int, 
                 max_text_len: int,
                 seed=None):
        self.news: pd.DataFrame = news
        self.behaviors: pd.DataFrame = behaviors
        self.tokenizer = tokenizer
        
        self.max_clicks = max_clicks
        self.max_text_len = max_text_len
        
        random.seed(seed)

        self.init_tokenization()

    def init_tokenization(self):
        news_enc = dict()
        for news_id, row in tqdm(self.news.iterrows(), total=len(self.news)):
            title, abstract = row['title'], row['abstract']
            news_enc[news_id] = \
                self.tokenizer(title, abstract, 
                               padding='max_length', 
                               max_length=self.max_text_len, 
                               truncation=True, 
                               return_tensors='pt')
        
        self.news_enc = news_enc
        
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        _, _, clicked_news, impressions = self.behaviors.iloc[idx]
        clicked_ids = clicked_news.split()
        candidate_ids, is_clicked = list(zip(*[imp.split('-') for imp in impressions.split()]))
        is_clicked = torch.tensor([int(b) for b in is_clicked], dtype=torch.float32)
        
        clicks_enc_list = [self.news_enc[k] for k in clicked_ids]
        
        short = self.max_clicks - len(clicked_ids)
        if short >= 0:
            # padding with all-zero attention masks
            clicks_enc_list.extend([
                dict(
                    input_ids=torch.ones(1, self.max_text_len, dtype=torch.int32), 
                    attention_mask=torch.zeros(1, self.max_text_len, dtype=torch.int32)
                ) \
                for _ in range(short)
            ])
        else:
            # randomly sample `max_clicks` news from all clicked news
            clicks_enc_list = random.sample(clicks_enc_list, self.max_clicks)

        clicks_enc = {
            'input_ids': torch.cat([enc['input_ids'] for enc in clicks_enc_list]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in clicks_enc_list])
        }
        
        candidates_enc_list = [self.news_enc[k] for k in candidate_ids]
        
        candidates_enc = {
            'input_ids': torch.cat([enc['input_ids'] for enc in candidates_enc_list]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in candidates_enc_list])
        }
        
        item = {
            'clicks': clicks_enc,
            'candidates': candidates_enc,
            'labels': is_clicked
        }
        
        return item

def gather_collate(batch):
        def _gather(x):
            if isinstance(x, dict):
                return {k: _gather(v) for k, v in x.items()}
            
            return x.view(-1, x.size(-1))
            
        batch = default_collate(batch)
        return _gather(batch)


class NewsGatherCollator(DataCollatorMixin):
    def __call__(self, features: List[Dict[str, Any]], return_tensors='pt') -> Dict[str, Any]:
        return gather_collate(features)