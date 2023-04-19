import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from utils.build_vocab import Vocabulary
import pickle

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(weights='DEFAULT')
        # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)  # {sequential: 9}
        # Linear(in_features=2048, out_features=256, bias=True)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # BatchNorm1d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):  # images {Tensor: (1, 3, 224, 224)}
        """Extract the image feature vectors."""
        features = self.resnet(images)  # {Tensor: (1, 2048, 1, 1)}
        features = Variable(features.data)  # {Tensor: (1, 2048, 1, 1)}
        # print(features.size(0))  # 1
        features = features.view(features.size(0), -1)  # {Tensor: (1, 2048)}
        features = self.bn(self.linear(features))  # {Tensor: (1, 256)}
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding(30, 256)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM(256, 512, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)  # Linear(in_features=512, out_features=30, bias=True)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        with open("./image_captioning/data/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 

            # print(hiddens.size())
            # print(states[0].size(),states[1].size())

            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]

            # print("stuff",type(predicted.data),predicted.data)
            # print(vocab.idx2word[1])
            # print("\nNNASDFKLASDJF\n\n",vocab.idx2word[predicted.data.cpu().numpy()[0]])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)

        # print("SAMPLED IDS",sampled_ids.size())
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()
