# 2023 Data Mining - Final Competition
## Framework
- News Encoder
    - conduct the task described in slide *p.8*
    - use the pooler(unfreezed) of freezed [RoBERTa](https://arxiv.org/abs/1907.11692)-base for news embedding
        - RoBERTa-base needs to be freezed since it's fine-tuning process is time- and memory-exhausted.
- User Clicks Encoder
    - conduct the task described in slide *p.9*
    - use [Self-Attention](https://arxiv.org/abs/1706.03762) mechanism to capture the relationship among the given news, and the first attention vector is taken for user representation.
- Recommendation System
    - conduct the whole inferrence process including the task described in slide *p.10*
    - receive clicked news and candidate news as input, and output the probabilities of the 15 candidate news

## File Description
- `train.py`: training process
- `model.py`: define the model classes
- `dataset.py`: define dataset
- `fix_data.ipynb`: fix the abnormal samples in `train_news.tsv`
