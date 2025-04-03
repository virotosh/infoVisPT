import glob
import os
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
tokenizer.train(files=paths, vocab_size=30_000, min_frequency=2,
                    limit_alphabet=1000, wordpieces_prefix='##',
                    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])


tokenizer.save_model('model')

tokenizer = BertTokenizer.from_pretrained('model/')

import torch
class Dataset(torch.utils.data.Dataset):
    """
    This class loads and preprocesses the given text data
    """
    def __init__(self, paths, tokenizer):
        """
        This function initialises the object. It takes the given paths and tokeniser.
        """
        # the last file might not have 10000 samples, which makes it difficult to get the total length of the ds
        self.paths = paths[:len(paths)-1]
        self.tokenizer = tokenizer
        self.data = self.read_file(self.paths[0])
        self.current_file = 1
        self.remaining = len(self.data)
        self.encodings = self.get_encodings(self.data)

    def __len__(self):
        """
        returns the lenght of the ds
        """
        return 10000*len(self.paths)
    
    def read_file(self, path):
        """
        reads a given file
        """
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        return lines

    def get_encodings(self, lines_all):
        """
        Creates encodings for a given text input
        """
        # tokenise all text 
        batch = self.tokenizer(lines_all, max_length=512, padding='max_length', truncation=True)

        # Ground Truth
        labels = torch.tensor(batch['input_ids'])
        # Attention Masks
        mask = torch.tensor(batch['attention_mask'])

        # Input to be masked
        input_ids = labels.detach().clone()
        rand = torch.rand(input_ids.shape)

        # with a probability of 15%, mask a given word, leave out CLS, SEP and PAD
        mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 2) * (input_ids != 3)
        # assign token 4 (=MASK)
        input_ids[mask_arr] = 4
        
        return {'input_ids':input_ids, 'attention_mask':mask, 'labels':labels}

    def __getitem__(self, i):
        """
        returns item i
        Note: do not use shuffling for this dataset
        """
        # if we have looked at all items in the file - take next
        if self.remaining == 0:
            self.data = self.read_file(self.paths[self.current_file])
            self.current_file += 1
            self.remaining = len(self.data)
            self.encodings = self.get_encodings(self.data)
        
        # if we are at the end of the dataset, start over again
        if self.current_file == len(self.paths):
            self.current_file = 0
                 
        self.remaining -= 1    
        return {key: tensor[i%10000] for key, tensor in self.encodings.items()}  




dataset = Dataset(paths = [str(x) for x in Path('data/').glob('**/*.txt')][:300], tokenizer=tokenizer)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8)

test_dataset = Dataset(paths = [str(x) for x in Path('data/').glob('**/*.txt')][300:], tokenizer=tokenizer)
valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)


from transformers import DistilBertForMaskedLM, DistilBertConfig

config = DistilBertConfig(
    vocab_size=30000,
    max_position_embeddings=514
)
model = DistilBertForMaskedLM(config)



from tqdm import tqdm
epochs = 10
optim = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda')

for epoch in range(epochs):
    loop = tqdm(train_loader, leave=True)
    
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
    print("Mean Test Loss", np.mean(losses))