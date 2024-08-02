import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14,train_CNN=False):
        super(Encoder, self).__init__()
        # load the pretrained ResNet-101 model
        self.enc_image_size = encoded_image_size
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        #resnet = models.resnet101(pretrained=False)
        # remove the classification head (the last two layers)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # set the model trainability
        self.fine_tune(train_CNN)

        # adaptive pool layer to resize the images to fixed size (encoded_image_size x encoded_image_size) regardless of their original size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

       

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


    def forward(self, images):
        # Images are of shape (batch_size, 3, image_size, image_size)
        out = self.resnet(images) # (batch_size, 2048, out_image_size, out_image_size)
        out = self.adaptive_pool(out) # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1) # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out  
    

class Attention(nn.Module):
    def __init__(self, encoded_image_dim, hidden_size, attention_dim):
        super(Attention, self).__init__()
        # the linear layer to transform the encoded image 
        self.encoder_att = nn.Linear(encoded_image_dim, attention_dim)
        # the linear layer to transform the hidden state of the decoder
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        # the linear layer to calculate values to be used in the attention mechanism
        self.full_att = nn.Linear(attention_dim, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # the attention scores should be calculated along the encoded image dimension

    def forward(self, image_embeddings,decoder_hidden):
        """
        :param image_embeddings: the encoded images from the encoder, shape: (batch_size, num_pixels, encoded_image_dim)
        :param decoder_hidden: the hidden state of the decoder (since we have a single LSTM cell, the hidden state is also the output), shape: (batch_size, hidden_size)
        """
        # project the encoded images
        att1 = self.encoder_att(image_embeddings) # (batch_size, num_pixels, attention_dim)
        # project the decoder hidden state
        att2 = self.decoder_att(decoder_hidden) # (batch_size, attention_dim)
        # add the two projections and apply the non-linearity
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2) # (batch_size, num_pixels)
        # calculate the attention weights
        alpha = self.softmax(att) # (batch_size, num_pixels)
        # calculate the context vector (the C vector is the multipliction of the attention scores and the image embeddings)
        context = (image_embeddings * alpha.unsqueeze(2)).sum(dim=1) # (batch_size, encoded_image_dim) (after multiplying the image encoding with the attention weights, we sum along the num_pixels dimension 
        return context, alpha
        

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, word_embedding_dim, hidden_size, vocab_size, encoded_img_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of the attention network
        :param word_embedding_dim: size of the word embeddings
        :param hidden_size: size of the hidden state of the RNN
        :param vocab_size: size of the vocabulary
        :param encoded_img_dim: feature size of the encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        # the parameters
        self.encoded_img_dim = encoded_img_dim
        self.attention_dim = attention_dim
        self.word_embedding_dim = word_embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        # attention network
        self.attention = Attention(encoded_img_dim, hidden_size, attention_dim)

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, word_embedding_dim)
        # dropout layer
        self.dropout = nn.Dropout(p=self.dropout)
        # Decoder LSTM Cell
        self.decode_step = nn.LSTMCell(word_embedding_dim + encoded_img_dim, hidden_size, bias=True) # the LSTM cell
        ## 2 linear layers to transform the encoded image feature vector to the hidden state dimension of the LSTM cell
        self.init_h = nn.Linear(encoded_img_dim, hidden_size) # to initialize the hidden state of the LSTM cell
        self.init_c = nn.Linear(encoded_img_dim, hidden_size) # to initialize the cell state of the LSTM cell
        # linear layer to create a sigmoid-activated gate (used in attention mechanism)
        self.f_beta = nn.Linear(hidden_size, encoded_img_dim) # to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        # linear layer to find scores over the vocabular
        self.fc = nn.Linear(hidden_size, vocab_size) # to find scores over the vocabulary
        
        ## initialize some layers with the uniform distribution
        # the embedding layer
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # the fc layer
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, image_embeddings):
        """
        :param image_embeddings: the encoded images from the encoder, of shape (batch_size, num_pixels = (enc_image_size * enc_image_size), encoded_img_dim)
        """
        mean_encoder_out = image_embeddings.mean(dim=1) # (batch_size, encoded_img_dim) this gets the average of all the pixels for each feature
        h = self.init_h(mean_encoder_out)  # (batch_size, hidden_size)
        c = self.init_c(mean_encoder_out)  # (batch_size, hidden_size)
        return h, c
    
    def forward(self, image_embeddings, captions, lengths):
        """
        :param image_embeddings: the encoded images from the encoder, of shape (batch_size, enc_image_size, enc_image_size, 2048 = encoded_img_dim)
        :param captions: the captions, of shape (batch_size, seq_length)
        :param lengths: the lengths of the captions, of shape (batch_size)
        """

        batch_size = image_embeddings.size(0)

        ## Flatten the image
        image_embeddings = image_embeddings.view(image_embeddings.size(0), -1, image_embeddings.size(-1)) # (batch_size, num_pixels = (enc_image_size * enc_image_size), encoded_img_dim)
        num_pixels = image_embeddings.size(1)

        ## sort the data by the length of the captions
        lengths, sorting_indices = lengths.sort(dim=0, descending=True) # this will return the lengths (sorted) and the indices that we will use to sort the rest
        image_embeddings = image_embeddings[sorting_indices]
        captions = captions[sorting_indices]


        ## get the word embeddings
        embeddings = self.embedding(captions) # (batch_size, seq_length, word_embedding_dim)

        ## initialize the hidden and cell states of the LSTM cell
        h, c = self.init_hidden_state(image_embeddings) # (batch_size, hidden_size)

        ## we wont process <EOS> token (why? because we don't need to predict anything after it)
        # so we will we will decrease the length of the captions by 1
        decode_lengths = (lengths - 1).tolist() # the decoding lengths will be the actual lengths - 1

        ## create tensors to hold the word prediction scores and alphas (attention weights)
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device) # will store the scores (the probabilities) of each word in the vocabulary for each time step (not all of them but to the maximum decoding length) for each example 
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device) # will store the attention weights for each pixel in the image for each time step (not all of them but to the maximum decoding length) for each example
        
        ## start the decoding process
        for t in range(max(decode_lengths)):
            # get the batch size at that time step (the number of examples that still didn't reach the <EOS> token)
            batch_size_t = sum([l > t for l in decode_lengths])
            # get the attention weights and the context vector (which is the multilication of the attention weights and the image embeddings then summed along the num_pixels dimension) 
            # (we will send the image embeddings and the hidden state of the LSTM cell up to the current batch size)
            context, alpha = self.attention(image_embeddings[:batch_size_t], h[:batch_size_t]) # context is of shape (batch_size_t, encoded_img_dim) and alpha=attention weights is of shape (batch_size_t, num_pixels)
            # gate: to control the amount of information that will be passed to the LSTM cell (shape of the gate is the same as the context vector)
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) # (batch_size_t, encoded_img_dim)
            # apply the gate to the context
            context = gate * context
            # concatenate the embeddings and the context vector
            concat_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1) # (batch_size_t, word_embedding_dim + encoded_img_dim)
            # pass the concatenated input to the LSTM cell
            h, c = self.decode_step(concat_input, (h[:batch_size_t], c[:batch_size_t])) # (batch_size_t, hidden_size)
            # pass the hidden state to the linear layer to get the scores
            preds = self.fc(self.dropout(h)) # (batch_size_t, vocab_size)
            # store the scores and the attention weights
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, captions, decode_lengths, alphas, sorting_indices 
    
