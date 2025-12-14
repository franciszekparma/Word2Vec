import  torch

from model import embds
from utils import TOPK_WORD, top_k, K, LOAD_CHECKPOINT, PATH_CHECHPOINT, LOWER_WORDS, DEVICE


if LOAD_CHECKPOINT:
  checkpoint = torch.load(PATH_CHECHPOINT, map_location=DEVICE, weights_only=False)
  embds = checkpoint['embeddings']

TOPK_WORD = TOPK_WORD.lower() if LOWER_WORDS else TOPK_WORD

print(f'\nThe {K} closest words to "{TOPK_WORD}" are:')
for (word, dist) in top_k(TOPK_WORD, embds, K):
  print(f"{word} -> {dist}")