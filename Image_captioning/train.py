import argparse
from tqdm import tqdm
import pickle
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.optim.lr_scheduler as lr_scheduler
transform = models.ResNet101_Weights.DEFAULT.transforms()
import sacrebleu
from dataset import *
from preprocess import *
from model import *
from utils import *



## Hyperparameters
# dataloader parameters
batch_size = 80
pin_memory = torch.cuda.is_available()
num_workers = 2

# model parameters
word_embedding_dim = 512
attention_dim = 512
hidden_size = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training parameters
epochs = 120
epochs_since_improvement = 0
encoder_lr = 1e-4 # learning rate for encoder if fine-tuning
decoder_lr = 4e-4 # learning rate for decoder
grad_clip = 5. # clip gradients at an absolute value of 
alpha_c = 1. # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0. # BLEU-4 score right now
fine_tune_encoder = False # fine-tune encoder?

# scheduler parameters
use_scheduler = False # use learning rate scheduler?
T_0 = 4  # number of epochs for the first restart
T_mult = 1  # no period length increase
eta_min = 1e-7  # minimum learning rate

# load model
load_model = False
checkpoint_path = None


def main():

    global best_bleu4, epochs_since_improvement, checkpoint_path
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the image directory')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary pickle file')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the training captions csv file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to the validation captions csv file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to the test captions csv file')

    args = parser.parse_args()

    img_paths = args.image_dir
    vocab_path = args.vocab_path
    train_csv = args.train_csv
    val_csv = args.val_csv
    test_csv = args.test_csv


    # load the vocabulary
    with open(vocab_path, "rb") as vocab_file:
        vocab = pickle.load(vocab_file)

    # load the dataframes
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    # initialize the dataset objects
    train_dataset = FlickrDataset(root_dir=img_paths, captions_df=train_df, vocabulary=vocab, transform=transform)
    val_dataset = FlickrDataset(root_dir=img_paths, captions_df=val_df, vocabulary=vocab, transform=transform, mode='val')
    test_dataset = FlickrDataset(root_dir=img_paths, captions_df=test_df, vocabulary=vocab, transform=transform, mode='test')


    # initialize the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=pin_memory, collate_fn=lambda batch: collate_fn(batch, vocab.stoi["<PAD>"]))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory, collate_fn=lambda batch: collate_fn(batch, vocab.stoi["<PAD>"]))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory, collate_fn=lambda batch: collate_fn(batch, vocab.stoi["<PAD>"]))


    # initialize the encoder and decoder
    decoder = DecoderWithAttention(attention_dim=attention_dim, word_embedding_dim=word_embedding_dim, hidden_size=hidden_size, vocab_size=len(vocab), dropout=dropout).to(device)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr)
    encoder = Encoder(train_CNN=fine_tune_encoder).to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)

    if use_scheduler:   
        decoder_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(decoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,verbose=True)
        encoder_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,verbose=True)

    # initialize the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"]) # we will ignore the loss whenever the target is a <PAD> token

    # load the model if specified
    if load_model:
        file_path = checkpoint_path
        checkpoint = torch.load(file_path)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        best_bleu4 = checkpoint['bleu-4']
        print("Model loaded with best BLEU-4 score of", best_bleu4)


    # start training
    for epoch in range(epochs):
        ## training phase 
        decoder.train()
        if fine_tune_encoder:
            encoder.train()

        # decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 4:
            break
        # if want to use step learning rate scheduler instead of cosine annealing uncomment the following lines
        #     if epochs_since_improvement > 0 and epochs_since_improvement % 2 == 0:
        #         adjust_learning_rate(decoder_optimizer, 0.9)
        #         if fine_tune_encoder:
        #             adjust_learning_rate(encoder_optimizer, 0.9)

        tk0 = tqdm(train_loader, total=len(train_loader), desc="Epoch {} Training".format(epoch+1))
        train_loss = 0
        train_examples = 0

        for i, (images, captions, lengths) in enumerate(tk0):
            # get the data to cuda if available
            images = images.to(device)
            captions = captions.to(device) 
            lengths = lengths.to(device)

            ## forward pass
            image_embeddings = encoder(images)
            logits, sorted_captions, decode_lengths, alphas, sorting_indices = decoder(image_embeddings, captions, lengths)

            # since we decoded starting with <SOS>, the predictions are all words after <SOS>, up to <EOS> (exclusive), so we need to adjust the targets also
            targets = sorted_captions[:, 1:]  # (batch_size, seq_length - <SOS>)


            ## Compute the cost and store it
            loss = criterion(logits.reshape(-1, logits.shape[2]), targets.reshape(-1)) # this reshapes the logits to be of shape (batch_size * seq_len, vocab_size) and the captions to be of shape (batch_size * seq_len) -we simply merged the batch_size and seq_len dimensions together for the criterion-
            train_loss += loss.item() * images.size(0)
            train_examples += images.size(0)

            # add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()



            ## Backward pass
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            ## clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            

            ## update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            ## update the progress bar
            tk0.set_postfix(loss=(train_loss / train_examples))

        ## validation phase
        decoder.eval()
        if fine_tune_encoder:
            encoder.eval()

        tk1 = tqdm(val_loader, total=len(val_loader), desc="Epoch {} Validation".format(epoch+1))
        val_loss = 0
        val_examples = 0
        bleu_score = 0

        
        with torch.no_grad():
            for i, (images, captions, lengths, all_captions) in enumerate(tk1):
                # get the data to cuda if available
                images = images.to(device)
                captions = captions.to(device)
                lengths = lengths.to(device)

                ## forward pass
                image_embeddings = encoder(images)
                logits, sorted_captions, decode_lengths, alphas, sorting_indices = decoder(image_embeddings, captions, lengths) 

                # since we decoded starting with <SOS>, the predictions are all words after <SOS>, up to <EOS> (exclusive), so we need to adjust the targets also
                targets = sorted_captions[:, 1:]

                ## Compute the cost and store it
                loss = criterion(logits.reshape(-1, logits.shape[2]), targets.reshape(-1))
                val_loss += loss.item() * images.size(0)
                val_examples += images.size(0)

                # add doubly stochastic attention regularization
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

                ## update the progress bar
                tk1.set_postfix(loss=(val_loss / val_examples))

                ## calculate the BLEU score
                # first, we need to sort the all_captions list based on the sorting_indices
                all_captions = [all_captions[j] for j in sorting_indices]
                # loop over the examples in the batch
                for j in range(logits.shape[0]):
                    example = logits[j] # shape (seq_len, vocab_size)
                    example = example.argmax(dim=1) # shape (seq_len)
                    example = [val_dataset.textualize(example)]
                    reference = all_captions[j]
                    score = sacrebleu.corpus_bleu(example, reference)
                    bleu_score += score.score


        avg_bleu_score = bleu_score / val_examples
        print(f"Epoch {epoch+1}, BLEU score: {avg_bleu_score}")
        if avg_bleu_score > best_bleu4:
            best_bleu4 = avg_bleu_score
            epochs_since_improvement = 0
            torch.save({
                'epoch': epoch,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'bleu-4': best_bleu4
            }, 'checkpoint.pth.tar')
        else:
            epochs_since_improvement += 1

        if use_scheduler:  
            decoder_scheduler.step()
            if fine_tune_encoder:
                encoder_scheduler.step()


if __name__ == "__main__":
    main()


