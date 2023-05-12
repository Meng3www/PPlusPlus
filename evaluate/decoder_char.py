# Decoder class for char-level models, for the use in get_accuracy_ts1.py

import torch.nn as nn
import torch


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, captions, hidden_prev):  ##
        """
        Decode image feature vectors and generates captions.
        """
        # print('captions: ', captions)  # device='cuda:0'
        embeddings = self.embed(captions)  # Tensor: (12, 256)
        # print('features.shape before cat: ', features.shape)  #
        # print('embeddings.shape before cat: ', embeddings.shape)  # torch.Size([12, 256])
        output, hidden = self.lstm(embeddings, hidden_prev)  ##
        # print('output shape: ', output.shape)  # torch.Size([12, 512])
        predictions = self.linear(output)
        # print('predictions shape: ', predictions.shape)  # torch.Size([12, 4987])

        return torch.nn.functional.log_softmax(predictions, dim=1), hidden  ##

