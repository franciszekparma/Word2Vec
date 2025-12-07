import torch
from torch import nn

from utils import EMB_DIM, DEVICE, EPOCHS
from data_prep import build_data
from train import train_model


class Word2Vec(nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, center_emb, y_embds):
    return center_emb @ y_embds.T


def main():
  vocab, all_words_in_sen = build_data()
  
  embds = {word: torch.randn((EMB_DIM), device=DEVICE, requires_grad=True) for word in vocab}
    
  model = Word2Vec().to(DEVICE)
  
  optimizer = torch.optim.AdamW(list(embds.values()), lr=1e-4, weight_decay=5e-3)
  loss_fn = nn.BCEWithLogitsLoss()
  
  train_model(
    vocab,
    all_words_in_sen,
    embds,
    model,
    optimizer,
    loss_fn,
    EPOCHS
    )
 
if __name__ == '__main__':
  main()