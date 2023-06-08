import random
from typing import Any, List, Dict, Callable
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch.utils.data import default_collate
from transformers.data.data_collator import DataCollatorMixin

class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 news_inputs: Dict[str, Any], 
                 behaviors: pd.DataFrame,
                 max_clicks: int, 
                 max_text_len: int,
                 seed=None):
        self.news_inputs = news_inputs
        self.behaviors: pd.DataFrame = behaviors
        
        self.max_clicks = max_clicks
        self.max_text_len = max_text_len
        
        random.seed(seed)
        
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        _, _, clicked_news, impressions = self.behaviors.iloc[idx]
        clicked_ids = clicked_news.split()
        candidate_ids, is_clicked = list(zip(*[imp.split('-') for imp in impressions.split()]))
        is_clicked = torch.tensor([int(b) for b in is_clicked], dtype=torch.float32)
        
        clicked_news_inputs = [self.news_inputs[k] for k in clicked_ids]
        
        short = self.max_clicks - len(clicked_ids)
        if short >= 0:
            # padding with all-zero attention masks
            clicked_news_inputs.extend([
                dict(
                    input_ids=torch.ones(1, self.max_text_len, dtype=torch.int32), 
                    attention_mask=torch.zeros(1, self.max_text_len, dtype=torch.int32)
                ) \
                for _ in range(short)
            ])
        else:
            # randomly sample `max_clicks` news from all clicked news
            clicked_news_inputs = random.sample(clicked_news_inputs, self.max_clicks)

        clicked_news_inputs = {
            'input_ids': torch.cat([enc['input_ids'] for enc in clicked_news_inputs]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in clicked_news_inputs])
        }
        
        candidate_news_inputs = [self.news_inputs[k] for k in candidate_ids]
        
        candidate_news_inputs = {
            'input_ids': torch.cat([enc['input_ids'] for enc in candidate_news_inputs]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in candidate_news_inputs])
        }
        
        item = {
            'clicks': clicked_news_inputs,
            'candidates': candidate_news_inputs,
            'labels': is_clicked
        }
        
        return item
    

class TestingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 news_inputs: Dict[str, Any], 
                 behaviors: pd.DataFrame,
                 max_clicks: int, 
                 max_text_len: int,
                 seed=None):
        self.news_inputs = news_inputs
        self.behaviors: pd.DataFrame = behaviors
        
        self.max_clicks = max_clicks
        self.max_text_len = max_text_len
        
        random.seed(seed)
        
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        _, _, clicked_news, impressions = self.behaviors.iloc[idx]
        clicked_ids = clicked_news.split()
        candidate_ids = impressions.split()
        
        clicked_news_inputs = [self.news_inputs[k] for k in clicked_ids]
        
        short = self.max_clicks - len(clicked_ids)
        if short >= 0:
            # padding with all-zero attention masks
            clicked_news_inputs.extend([
                dict(
                    input_ids=torch.ones(1, self.max_text_len, dtype=torch.int32), 
                    attention_mask=torch.zeros(1, self.max_text_len, dtype=torch.int32)
                ) \
                for _ in range(short)
            ])
        else:
            # randomly sample `max_clicks` news from all clicked news
            clicked_news_inputs = random.sample(clicked_news_inputs, self.max_clicks)

        clicked_news_inputs = {
            'input_ids': torch.cat([enc['input_ids'] for enc in clicked_news_inputs]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in clicked_news_inputs])
        }
        
        candidate_news_inputs = [self.news_inputs[k] for k in candidate_ids]
        
        candidate_news_inputs = {
            'input_ids': torch.cat([enc['input_ids'] for enc in candidate_news_inputs]), 
            'attention_mask': torch.cat([enc['attention_mask'] for enc in candidate_news_inputs])
        }
        
        item = {
            'clicks': clicked_news_inputs,
            'candidates': candidate_news_inputs
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