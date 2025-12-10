from datasets import Dataset, load_dataset
import string
import os

from utils import LOWER_WORDS, SHOW_DATA_STATS

def build_data(path="data", show_stats=SHOW_DATA_STATS): 
  if not os.path.exists(path):
    dataset = load_dataset("embedding-data/flickr30k_captions_quintets")
    dataset.save_to_disk(path)
  
  print(f"Loading data...")
  dataset = Dataset.from_file("data/train/data-00000-of-00001.arrow")

  all_sentences = []
  for example in dataset:
    all_sentences.extend(example['set'])    

  all_words_in_sen = []
  all_words = []
  for sentence in all_sentences:
    sentence = sentence.translate(str.maketrans('', '', string.punctuation)) #removes punctuaction and most of the special characters
    words_in_sen = sentence.split()
    if LOWER_WORDS:
      all_words_in_sen.append([word.lower() for word in words_in_sen])  
      all_words.extend([word.lower() for word in words_in_sen])
    else:
      all_words_in_sen.append([word for word in words_in_sen])  
      all_words.extend([word for word in words_in_sen])

  if LOWER_WORDS:
    vocab = list(set([word.lower() for word in all_words])) #getting only the unique words (no duplicates)
  else:
    vocab = list(set(all_words))
  vocab = sorted(vocab) #for reproducibility
  
  if show_stats:
    print(f"Total sentences: {len(all_sentences)}")
    print(f"Total words (with duplicates): {len(all_words)}")
    print(f"Vocab size: {len(vocab)}")
  
  return vocab, all_words_in_sen
