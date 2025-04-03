from tqdm.auto import tqdm
from datasets import load_dataset
import os
import pandas as pd
import random

ds = load_dataset("openwebtext")


# save text in chunks of 10000 samples
text = []
ind = 0

for sample in tqdm(ds['train']):
    # replace all newlines
    sample = sample['text'].replace('\n','')
    
    # append cleaned sample to all texts
    text.append(sample)
    
    # if we processed 10000 samples, write them to a file and start over
    if len(text) == 10000:
        with open(f"data/text_{ind}.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(text))
        text = []
        ind += 1

# write remaining samples to a file
with open(f"data/text_{i}.txt", 'w', encoding='utf-8') as f:
    f.write('\\n'.join(text))