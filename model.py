import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput

class NewsEncoder(nn.Module):
    def __init__(self, backbone, output_dim):
        """An encoder module to encode the words in news.
        
        Args:
            backbone: Bert or RoBerta built from huggingface
            output_dim: the dimension of the encoding vector
        """
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim
        
        self.fc = nn.Linear(self.backbone.config.hidden_size, output_dim)
        
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids, attention_mask).pooler_output  # dimension default in 768
        h = self.fc(h)
        
        return h

    
class UserClicksEncoder(nn.Module):
    # Here we utilize the multi-head self-attention mechanism to infer the relationship between clicked news
    def __init__(self, news_vec_dim, user_vec_dim, attn_num_heads, attn_dropout=0.):
        super().__init__()
        self.news_vec_dim = news_vec_dim
        self.user_vec_dim = user_vec_dim
        
        self.self_attn = nn.MultiheadAttention(embed_dim=user_vec_dim, 
                                               num_heads=attn_num_heads, 
                                               dropout=attn_dropout, 
                                               batch_first=True)
        
    def forward(self, X):
        # X: (B, M+1, D) with a special token vector for user vector encoding
        H = self.self_attn(X, X, X, need_weights=False)[0]
        H = H[:, [0], :]  # take the vector at the first pos for encoding
        
        return H
    
class RecommendationSystem(nn.Module):
    def __init__(self, news_encoder: NewsEncoder, user_encoder: UserClicksEncoder, n_labels: int, max_clicks: int):
        super().__init__()
        self.news_encoder: NewsEncoder = news_encoder
        self.user_encoder: UserClicksEncoder = user_encoder
        
        self.n_labels = n_labels
        self.max_clicks = max_clicks
        
        self.user_token_embed = nn.Parameter(torch.zeros(1, self.user_encoder.news_vec_dim, requires_grad=True))  # trainable variable
        
    def forward(self, clicks, candidates, labels):
        clicks_vecs = self.news_encoder(**clicks)
        clicks_vecs = clicks_vecs.view(-1, self.max_clicks, clicks_vecs.size(-1))  # divide to BxMxD
        
        embedding = self.user_token_embed.expand(clicks_vecs.size(0), *self.user_token_embed.size()).contiguous()
        clicks_vecs = torch.cat([embedding, clicks_vecs], dim=1)
        user_vec = self.user_encoder(clicks_vecs)  # Bx1xD
        
        candidates_vecs = self.news_encoder(**candidates)
        candidates_vecs = candidates_vecs.view(-1, self.n_labels, candidates_vecs.size(-1))  # divide to Bx15xD
        
        outputs = torch.matmul(candidates_vecs, user_vec.transpose(1, 2)).view(-1, self.n_labels)  # Bx15
        
        loss = None
        if labels is not None:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs.view(-1, self.n_labels), labels)
        
        return SequenceClassifierOutput(loss=loss, logits=outputs)