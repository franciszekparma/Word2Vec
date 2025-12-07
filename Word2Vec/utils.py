import torch
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOWER_WORDS = True
SHOW_DATA_STATS = False

EMB_DIM = 32
WINDOW = 2

EPOCHS=32

def sample_negatives(vocab, n, context_words, center_word):
  neg_vocab = vocab.copy()
  
  context_words = list(set(context_words))
  
  neg_vocab.remove(center_word)
  for word in context_words:
    if word in neg_vocab:
      neg_vocab.remove(word)
  
  vocab_keys = neg_vocab
  vocab_len = len(vocab_keys)
  
  negatives = []
  for i in range(n):
    negatives.append(vocab_keys[random.randint(0, vocab_len - 1)])
    
  return negatives