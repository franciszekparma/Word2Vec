from datasets import load_dataset, load_from_disk


dataset = load_dataset("embedding-data/flickr30k_captions_quintets")

dataset.save_to_disk("dataset/flickr30k_captions")