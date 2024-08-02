import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
from PIL import Image
from torchvision import models
transform = models.ResNet101_Weights.DEFAULT.transforms()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import argparse
import pickle
import matplotlib.cm as cm
from model import *
from preprocess import *


# model parameters
word_embedding_dim = 512
attention_dim = 512
hidden_size = 512
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tune_encoder = False # fine-tune encoder?


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=10):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """
    encoder.eval()
    decoder.eval()
    k = beam_size
    vocab_size = len(word_map)

    # load the image
    image = Image.open(image_path).convert("RGB")
    plt.imshow(image)
    # transform the image
    image = transform(image).unsqueeze(0)
    # get the image to the device
    image = image.to(device)
    
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
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

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1) # shape: (s, vocab_size)

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

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<EOS>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]
    # get the words
    # caption = val_dataset.textualize(torch.tensor(seq))
    # print(caption)
    alphas = torch.FloatTensor(alphas)

    encoder.train()
    decoder.train()
    return seq, alphas

def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True, output_path=None):
    """
    Visualizes caption with weights at every word and optionally saves the plot.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    :param save_path: path to save the plot
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]
    # Adjust the figure size here
    plt.figure(figsize=(20, 10))  # You can change the width and height as needed
    
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(int(np.ceil(len(words) / 5.)), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    if output_path:
        plt.savefig(os.path.join(output_path,image_path.split('/')[-1]), bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # parse the mode (image, directory) and the model path and the vocabulary path
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='Mode: image or directory')
    parser.add_argument('--input_path',required=True, type=str, help='Path to the image or directory')
    parser.add_argument('--vocab_path',required=True, type=str, help='Path to the vocabulary')
    parser.add_argument('--checkpoint_path',required=True, type=str, help='Path to the model')
    parser.add_argument('--output_dir',required=True, type=str, help='Path to save the images with the captions')
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for Captioning')

    args = parser.parse_args()

    mode = args.mode
    image_dir_path = args.input_path
    vocab_path = args.vocab_path
    model_path = args.checkpoint_path
    output_path = args.output_dir
    beam_size = args.beam_size

    # load the vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # create the encoder and decoder
    decoder = DecoderWithAttention(attention_dim=attention_dim, word_embedding_dim=word_embedding_dim, hidden_size=hidden_size, vocab_size=len(vocab), dropout=dropout).to(device)
    encoder = Encoder(train_CNN=fine_tune_encoder).to(device)

    # load the checkpoint
    checkpoint = torch.load(model_path)
    decoder.load_state_dict(checkpoint['decoder'])
    encoder.load_state_dict(checkpoint['encoder'])
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    best_bleu4 = checkpoint['bleu-4']
    print(f"Loded checkpoint with BLEU-4 score: {best_bleu4}")

    images = []
    if mode == 'image':
        images.append(image_dir_path)
    elif mode == 'directory':
        images = os.listdir(image_dir_path)
        images = [os.path.join(image_dir_path, image) for image in images]

    for image_path in images:
        seq, alphas = caption_image_beam_search(encoder, decoder, image_path, vocab.stoi, beam_size=beam_size)
        visualize_att(image_path, seq, alphas, vocab.itos, output_path=output_path)


    