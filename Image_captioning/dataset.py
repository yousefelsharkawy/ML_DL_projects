import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pad_sequence


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_df, vocabulary, transform=None, mode = 'train'):
        # root_dir: the directory where the images are stored
        self.root_dir = root_dir
        # captions_df: the dataframe containing the image names and their captions
        self.df = captions_df
        self.image_names = self.df['image']
        self.captions = self.df['caption']
        self.transform = transform
        # the vocabulary object
        self.vocab = vocabulary
        self.mode = mode

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # get the image 
        img_name = self.image_names[idx]
        image = Image.open(os.path.join(self.root_dir, img_name)).convert("RGB")
        #plt.imshow(image)
        if self.transform:
            image = self.transform(image)

        # get the caption and numericalize it
        caption = self.captions[idx]
        # first we add the index of the <SOS> token 
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        # then we add the index of each token in the caption
        numericalized_caption += self.vocab.numericalize(caption)
        # finally we add the index of the <EOS> token
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # we also return the length of the caption
        caption_length = len(numericalized_caption)

        # if the mode is train, return the image and the caption
        if self.mode == 'train':
            return image, torch.tensor(numericalized_caption) , caption_length
        # if the mode is val or test, return the image, the caption and all captions for that image
        else:
            all_captions = self.df[self.df['image'] == img_name]['caption'].tolist()
            all_captions = [[caption.lower()] for caption in all_captions]
            return image, torch.tensor(numericalized_caption), caption_length, all_captions
        
        
    def textualize(self, tensor):
        # convert a tensor of indices to the corresponding text if it is not a special token
        sentence = ""
        add_space = True
        for i in tensor:
            i = i.item()
            if i == self.vocab.stoi["<SOS>"] or i == self.vocab.stoi["<EOS>"] or i == self.vocab.stoi["<PAD>"] or i == self.vocab.stoi["<UNK>"]:
                continue
            # if the word is "-" we don't add a space before it or after it
            if self.vocab.itos[i] == "-":
                sentence += self.vocab.itos[i]
                add_space = False
            else:
                if add_space:
                    sentence += " " + self.vocab.itos[i]
                else:
                    sentence += self.vocab.itos[i]
                    add_space = True
        return sentence
    
def collate_fn(batch, pad_idx):
    # batch is a list of all examples in the batch, in each example we have a tuple (img, caption) -from the dataset implementation-
    # pad_idx is the index of the <PAD> token in the vocabulary, which is basically the value we will use to pad the sequences
    images = [item[0].unsqueeze(0) for item in batch] # unsqueeze to add a dimension at the beginning of the tensor, so it will be (1, 3, 224, 224) instead of (3, 224, 224), this dimension will be used to concatenate the images
    images = torch.cat(images, dim=0) # concatenate the images along the first dimension    
    captions = [item[1] for item in batch] # we didn't unsqueeze here because pad_sequence will do that for us
    captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx) # pad the sequences
    # caption lengths
    lengths = torch.LongTensor([item[2] for item in batch])


    if len(batch[0]) == 3:
        return images, captions.long(), lengths
    else:
        all_captions = [item[3] for item in batch]
        return images, captions.long(), lengths, all_captions
