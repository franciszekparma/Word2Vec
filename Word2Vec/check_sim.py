import  torch

from model import embds
from utils import WORD_A, WORD_B, euc_dist, cosine_sim, LOAD_EMBDS, PATH_CHECHPOINT


if LOAD_EMBDS:
  checkpoint = torch.load(PATH_CHECHPOINT, map_location='cpu', weights_only=False)
  emb_a = checkpoint['embeddings'][WORD_A].unsqueeze(0)
  emb_b = checkpoint['embeddings'][WORD_B].unsqueeze(0)
else:
  emb_a = embds[WORD_A].unsqueeze(0)
  emb_b = embds[WORD_B].unsqueeze(0)

print(f'Euclidean distance between "{WORD_A}" and "{WORD_B}": {euc_dist(emb_a, emb_b)}')
print(f'Cosine Similarity between "{WORD_A}" and "{WORD_B}": {cosine_sim(emb_a, emb_b)}')