from pathlib import Path
import torch
import time
from pathlib import Path
from transformers import DistilBertTokenizerFast
import os
from transformers import DistilBertConfig
from transformers import DistilBertForMaskedLM
from tokenizers import BertWordPieceTokenizer
from tqdm.auto import tqdm
from torch.optim import AdamW
import torchtest
from transformers import pipeline


from distilbert import test_model
from distilbert import Dataset

import numpy as np

import glob
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from pathlib import Path
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

paths = [str(x) for x in Path('data').glob('**/*.txt')]

tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=True
)
tokenizer.train(files=paths, vocab_size=30_000, min_frequency=1,
                    limit_alphabet=1000, wordpieces_prefix='##',
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])


tokenizer.save_model('mytokeniser')

tokenizer = DistilBertTokenizerFast.from_pretrained('mytokeniser', max_len=512)
tokenizer.save_pretrained("distilbert_tokenizer")

assert len(tokenizer.vocab) == 30_000




dataset = Dataset(paths = [str(x) for x in Path('data/').glob('**/*.txt')][:300], tokenizer=tokenizer)
loader = torch.utils.data.DataLoader(dataset, batch_size=8)

test_dataset = Dataset(paths = [str(x) for x in Path('data/').glob('**/*.txt')][300:], tokenizer=tokenizer)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

i = iter(dataset)

for j in range(10):
    sample = next(i)
    
    input_ids = sample['input_ids']
    attention_masks = sample['attention_mask']
    labels = sample['labels']
    
    # check if the dimensions are right
    assert input_ids.shape[0] == (512)
    assert attention_masks.shape[0] == (512)
    assert labels.shape[0] == (512)
    
    # if the input ids are not masked, the labels are the same as the input ids
    assert np.array_equal(input_ids[input_ids != 4].numpy(),labels[input_ids != 4].numpy())
    # input ids are zero if the attention masks are zero
    assert np.all(input_ids[attention_masks == 0].numpy()==0)
    # check if input contains masked tokens (we can't guarantee this 100% but this will apply) most likely
    assert np.any(input_ids.numpy() == 4)
print("Passed")

config = DistilBertConfig(
    vocab_size=30000,
    max_position_embeddings=514
)
model = DistilBertForMaskedLM(config)


device = torch.device('cuda')

model.to(device)




# get smaller dataset
test_ds = Dataset(paths = [str(x) for x in Path('data/').glob('**/*.txt')][:2], tokenizer=tokenizer)
test_ds_loader = torch.utils.data.DataLoader(test_ds, batch_size=2)
optim=torch.optim.Adam(model.parameters())


test_model(model, optim, test_ds_loader, device)

# we use AdamW as the optimiser
optim = AdamW(model.parameters(), lr=1e-4)


epochs = 1

for epoch in range(epochs):
    loop = tqdm(loader, leave=True)
    
    # set model to training mode
    model.train()
    losses = []
    
    # iterate over dataset
    for batch in loop:
        optim.zero_grad()
        
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        loss.backward()
        
        optim.step()
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
        
    print("Mean Training Loss", np.mean(losses))
    losses = []
    loop = tqdm(test_loader, leave=True)
    
    # set model to evaluation mode
    model.eval()
    
    # iterate over dataset
    for batch in loop:
        # copy input to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # predict
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # update weights
        loss = outputs.loss
        
        # output current loss
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())
        
        del input_ids
        del attention_mask
        del labels
    print("Mean Test Loss", np.mean(losses))