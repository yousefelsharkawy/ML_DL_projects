import spacy
spacy_eng = spacy.load("en_core_web_sm")
import argparse
import pandas as pd
import pickle
import numpy as np

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    # we can impplement this to return the length of the vocabulary
    def __len__(self):
        return len(self.itos)

    @staticmethod # this is a static method, which means it will be called on the class itself and won't need to access any instance data (no self)
    def tokenizer_eng(text):
        # mine: we could convert the text to lower case before tokenization
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    # this method will take a list of all the text examples (captions) and then builds the vocabulary
    def build_vocabulary(self, sentence_list):
        frequencies = {} # this will store the frequency of each word in the dataset, so that we decide whether to keep it or not
        idx = 4 # we start the index from 4, because 0,1,2,3 are already taken by the special tokens
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1
                # add the word to the vocabulary if it ever reaches the frequency threshold (==)
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    # this function takes any text and calls stoi on each token in the text
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]
    

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset: flickr8k or flickr30k')
    parser.add_argument('--captions_path', type=str,required=True, help='Path to the captions.txt file')
    parser.add_argument('--frequency_threshold', type=int, required=True, help='Frequency threshold for creating the vocabulary')

    args = parser.parse_args()

    chosen_dataset = args.dataset_name
    captions_path = args.captions_path
    frequency_threshold = args.frequency_threshold

    # read the captions file
    df = pd.read_csv(captions_path)
    # drop any rows with missing values
    df.dropna(inplace=True)

    captions = df['caption']
    # initialize the vocabulary object
    vocab = Vocabulary(freq_threshold=frequency_threshold)
    # build the vocabulary using the captions list
    vocab.build_vocabulary(captions.tolist())

    print(f"Length of the vocabulary of frequency threshold {frequency_threshold} is {len(vocab)}")

    # save the vocabulary to a file
    with open(f"{chosen_dataset}_vocab.pkl", "wb") as vocab_file:
        pickle.dump(vocab, vocab_file)
        print(f"Vocabulary saved to {chosen_dataset}_vocab.pkl")

    # Split the dataset into training, validation and test sets
    images = df['image'].unique()

    # shuffle the images
    np.random.seed(0)
    images = np.random.permutation(images)

    if chosen_dataset == 'flickr8k':
        # split the images into training, validation and test sets
        train_images = images[:6091]
        val_images = images[6091:7091]
        test_images = images[7091:]
    elif chosen_dataset == 'flickr30k':
        train_images = images[:28783]
        val_images = images[28783:30783]
        test_images = images[30783:]

    # construct the dataframes
    train_df = df[df['image'].isin(train_images)].reset_index(drop=True)
    val_df = df[df['image'].isin(val_images)].reset_index(drop=True)
    test_df = df[df['image'].isin(test_images)].reset_index(drop=True)


    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # save the dataframes to files
    train_df.to_csv(f"{chosen_dataset}_train.csv", index=False)
    val_df.to_csv(f"{chosen_dataset}_val.csv", index=False)
    test_df.to_csv(f"{chosen_dataset}_test.csv", index=False)


if __name__ == "__main__":
    main()

    

