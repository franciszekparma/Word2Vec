from datasets import Dataset, load_dataset
import pyarrow as pa


dataset = load_dataset("embedding-data/flickr30k_captions_quintets")
dataset.save_to_disk("dataset/flickr30k_captions")
dataset = Dataset.from_file("dataset/flickr30k_captions/train/data-00000-of-00001.arrow")

all_sentences = []
for example in dataset:
  all_sentences.extend(example['set'])
  
all_words = []
for sentence in all_sentences:
  words_in_sen = sentence.split()
  all_words.extend(words_in_sen)
  
vocab = list(set(all_words)) #getting only the unique words (no duplicates)

print(f"Total sentences: {len(all_sentences)}")
print(f"Total words (with duplicates): {len(all_words)}")
print(f"Vocab size: {len(vocab)}")