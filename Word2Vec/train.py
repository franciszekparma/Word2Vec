import torch
import numpy as np
from tqdm.auto import tqdm

from utils import sample_negatives, WINDOW, DEVICE


def train_model(
  vocab, 
  all_sentences, 
  all_words_in_sen, 
  target_embds, 
  contex_embds, 
  model, 
  loss_fn, 
  optimizer,
  epochs
  ):
  
  for epoch in tqdm(range(epochs)):
    model.train()
    
    losses = []
    
    for s, sentence in enumerate(all_sentences):
      for w_l, word_list in enumerate(all_words_in_sen):
        for w, word in enumerate(word_list):    
          y_pos_words = word_list[max(0, w - WINDOW) : w] + word_list[w + 1 : w + WINDOW + 1] #the words that we want the target to be similar to
          y_pos_embds = [contex_embds[x] for x in y_pos_words]
          y_pos_labels = [1 for x in range(len(y_pos_words))]
          
          y_neg_words = sample_negatives(vocab, n=2*WINDOW, context_words=y_pos_words, center_word=word)
          y_neg_embds = [contex_embds[x] for x in y_neg_words]
          y_neg_labels = [0 for x in range(len(y_neg_words))]
          
          y_embds = torch.tensor(y_pos_embds + y_neg_embds, device=DEVICE, requires_grad=True, pin_memory=True)
          y_labels = torch.tensor(y_pos_labels + y_neg_labels, device=DEVICE)
        
          center_word = word
          center_emb = target_embds[center_word] #the word embedding that we are currently at
          
          
          y_preds = model(center_emb, y_embds)
          
          loss = loss_fn(y_preds, y_labels)
          losses.append(loss.item())
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      
    print(f"Epoch: {epoch}")
    print(f"Loss: {np.mean(np.array(losses)):.4f}")