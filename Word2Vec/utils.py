import torch
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 24

LOWER_WORDS = True
SHOW_DATA_STATS = False

EMB_DIM = 128
WINDOW = 4

EPOCHS = 32

WORD_A = "hard"
WORD_B = "work"

random.seed(SEED)

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

def euc_dist(emb1, emb2):
  return ((torch.sum((emb1 - emb2)**2))**0.5).item()

def cosine_sim(emb1, emb2):
  return ((emb1 @ emb2.T) / (torch.norm(emb1) * torch.norm(emb2))).item()
