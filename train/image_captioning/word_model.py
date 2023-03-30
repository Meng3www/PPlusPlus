import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from utils.build_vocab import Vocabulary
import pickle


# getting data

# data cleaning

# getting the feature vector from images

# load data for training

# tokenise
	# utils/build_vocab.py

# CNN-RNN model
	# Feature: train/image_captioning/char_model.py
	# Sequence
	# Decoder
# training
	# train/Model.py



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
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
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # embeddings – padded batch of variable length sequences.
        # lengths – list of sequence lengths of each batch element (must be on the CPU if provided as a tensor)

        ###################### what is `length`?
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        ###################### what is the output? output, (hidden, cell) or hiddens, _?
        # hiddens, _ = self.lstm(packed)
        output, (hidden, cell) = self.lstm(embeddings)

        outputs = self.linear(hiddens[0])
        return outputs
        ##################### 06c-char-level-LSTM.py
        # output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim=1))
        # predictions = self.linear_map(output)
        # return torch.nn.functional.log_softmax(predictions, dim=1), hidden
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        with open("data/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)  # a Vocabulary() from utils/build_vocab.py
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

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

def train(features, captions):
    # get a fresh hidden layer
    hidden = lstm.initHidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence
    predictions, hidden = lstm(features, captions, hidden)
    # compute loss (NLLH)
    ##################### what is the correct argument captions[1:len(name)]?
    loss = criterion(predictions[:-1], captions[1:len(name)])
    # perform backward pass
    loss.backward()
    # perform optimization
    optimizer.step()
    # return prediction and loss
    return loss.item()


if __name__ == '__main__':

    # temp_dict = ["a", "list", "of", "words"]
    # with open("test/test.pkl", 'wb') as f:
    #     pickle.dump(temp_dict, f)
    # with open("test/test.pkl", 'rb') as f:
    #     vocab = pickle.load(f)
    # print(vocab)

    lstm = DecoderRNN(embed_size=0, hidden_size=0, vocab_size=0, num_layers=0)
                # cat_embedding_size=32, n_cat=n_categories, ####### features
                # char_embedding_size=embed_size,
                # n_char=vocab_size,
                # output_size=vocab_size,
    # training objective
    criterion = nn.NLLLoss(reduction='sum')
    # learning rate
    learning_rate = 0.005
    # optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # training parameters
    n_iters = 50000  #######?
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # will be reset every 'plot_every' iterations

    # start = time.time()

    for iter in range(1, n_iters + 1):
        loss = train(features=0, captions=0)
        total_loss += loss

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
            print(all_losses)

