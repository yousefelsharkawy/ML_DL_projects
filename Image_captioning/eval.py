from tqdm import tqdm
import torch
import torch.nn.functional as F
import sacrebleu
from torch.utils.data import DataLoader
import torchvision.models as models
transform = models.ResNet101_Weights.DEFAULT.transforms()
import argparse
import pickle
import pandas as pd
from model import *
from preprocess import *
from dataset import *


def calculate_bleu4_score(encoder,decoder,beam_size,loader):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    decoder.eval()
    encoder.eval()
    total_images = len(loader)
    total_bleu_score = 0

    # For each image
    for i, (img, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        img = img.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder(img)  # (1, enc_image_size, enc_image_size, encoder_dim)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<SOS>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<EOS>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
        if len(complete_seqs_scores) == 0:
            total_bleu_score += 0
            continue
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        example = dataset.textualize(torch.tensor(seq))
        caption = dataset.textualize(caps[0])
#         print(example)
#         print(caption)
#         print(allcaps[0])
        
        # Calculate BLEU-4 score using sacrebleu
        example_score = sacrebleu.corpus_bleu([example], allcaps[0]).score
        caption_score = sacrebleu.corpus_bleu([caption], allcaps[0]).score
        #print(example_score)
        #print(caption_score)
        if caption_score == 0:
            #print("0 caption")
            total_images -=1
            continue
        #score = (example_score / caption_score) * 100
        total_bleu_score += example_score

    encoder.train()
    decoder.train()
    
    avg_bleu_score = total_bleu_score / total_images
    return avg_bleu_score


# model parameters
word_embedding_dim = 512
attention_dim = 512
hidden_size = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tune_encoder = False # fine-tune encoder?



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse the vocabulary path, checkpoint path, 
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the vocabulary file')
    parser.add_argument('--checkpoint_path', type=str,required=True, help='Path to the checkpoint file')
    parser.add_argument('--imgs_path', type=str,required=True, help='Path to the images')
    parser.add_argument('--df_path', type=str,required=True, help='Path to the dataframe you want to evaluate on')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for evaluation')
    
    args = parser.parse_args()

    vocab_path = args.vocab_path
    checkpoint_path = args.checkpoint_path
    img_paths = args.imgs_path
    df_path = args.df_path
    beam_size = args.beam_size

    # load the vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    # load the model
    decoder = DecoderWithAttention(attention_dim=attention_dim, word_embedding_dim=word_embedding_dim, hidden_size=hidden_size, vocab_size=len(vocab), dropout=dropout).to(device)
    encoder = Encoder(train_CNN=fine_tune_encoder).to(device)

    # load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.load_state_dict(checkpoint['encoder'])
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    best_bleu4 = checkpoint['bleu-4']
    print(f"Loded checkpoint with BLEU-4 score: {best_bleu4}")

    

    # Load the dataset
    df = pd.read_csv(df_path)
    word_map = vocab.stoi
    vocab_size = len(vocab)
    
    
    dataset = FlickrDataset(root_dir=img_paths, captions_df=df ,vocabulary=vocab ,transform=transform, mode='val')

    # craete data loaders with batch_size = 1 (Necessary for evaluation)
    loader = DataLoader(dataset, batch_size=1,shuffle=False, collate_fn=lambda batch: collate_fn(batch, vocab.stoi["<PAD>"]))
    
    score = calculate_bleu4_score(encoder,decoder,beam_size=beam_size,loader=loader)
    print(f"BLEU-4 score at beam size {beam_size}: {score}")