import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from utils.build_vocab import Vocabulary
import pickle
from os.path import isfile
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable


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
        resnet = models.resnet152(weights='DEFAULT')  # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)  # Linear(in_features=2048, out_features=256, bias=True)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)  # BatchNorm1d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):  # return from load_image() in utils/sample.py
        """Extract the image feature vectors."""
        features = self.resnet(images)  # {Tensor: (1, 2048, 1, 1)}
        features = Variable(features.data)  # {Tensor: (1, 2048, 1, 1)}
        features = features.view(features.size(0), -1)  # {Tensor: (1, 2048)}
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # self.cat_embedding = nn.Embedding(n_cat, cat_embedding_size)  # (18, 32)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # output, (h_n, c_n), input_size = cat_embedding_size+char_embedding_size (32+32=64)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions):
        """
        Decode image feature vectors and generates captions.
        features: vector
        captions: tensor
        lengths: hidden?
        """
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # embeddings – padded batch of variable length sequences.
        # lengths – list of sequence lengths of each batch element (must be on the CPU if provided as a tensor)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # output, (h_n, c_n)
        # hiddens, _ = self.lstm(packed)
        # output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim=1))
        # Defaults to zeros if (h_0, c_0) is not provided.
        output, _ = self.lstm(embeddings)
        # get LSTM outputs
        # lstm_output, (h, c) = self.lstm(x, hidden)

        predictions = self.linear(output)
        return torch.nn.functional.log_softmax(predictions, dim=1)

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


embed_size=256
hidden_size=512
num_layers=1


def getTrainingPair():
    """
    get training item in desired input format (vectors of indices)
    :return:
    features: vector after pre-trained CNN
    captions: to be used in embeddings = self.embed(captions)
    """
    #todo:
    cnn = EncoderCNN(embed_size)
    cnn.eval()
    # self.encoder.load_state_dict(torch.load(self.encoder_path, map_location={'cuda:0': 'cpu'}))
    # 		if torch.cuda.is_available():
    # 			self.encoder.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    json_data = json.loads(open('vg_data/region_descriptions.json', 'r').read())
    with open("vg_data/vocab_small.pkl", 'rb') as f:
        vocab = pickle.load(f)  # a Vocabulary() from utils/build_vocab.py
        # vocab.idx2word or a list of index?

    for each_dict in json_data:
        # vg_data/VG_100K_2/71.jpg
        file_path = "vg_data/VG_100K_2/" + str(each_dict["id"]) + ".jpg"
        if isfile(file_path):
            # the picture exists
            im = Image.open(file_path)
            # each im contains a list of regions
            reg_list = each_dict['regions']
            for each_region in reg_list:
                # {"region_id": int, "width": int, "height": int, "image_id": int, "phrase": str, "y": int, "x": int}
                left = each_region["x"]  # x
                top = each_region["y"]  # y
                right = each_region["x"] + each_region["width"]  # x+width
                bottom = each_region["y"] + each_region["height"]  # y+height
                im1 = im.crop((left, top, right, bottom))
                tokens = each_region['phrase'].lower().split(' ')
                # im1.show()
                # resize
                im1 = im1.resize([224, 224], Image.Resampling.LANCZOS)
                # transform (1, 3, 224, 224)
                im1 = transform(im1).unsqueeze(0)
                # feed into cnn
                # self.encoder(to_var(load_image(url, self.transform)))
                # to_var() from utils/sample.py
                if torch.cuda.is_available():
                    im1 = im1.cuda()
                features = cnn(Variable(im1))  # cnn.forward(Variable(im1))

                # read captions, get index
                captions = [vocab.word2idx["<start>"]]
                # + [vocab.word2idx[word] for word in tokens]]
                len_token = 0
                for word in tokens:
                    if word in vocab.word2idx:
                        captions.append(vocab.word2idx[word])
                    else:
                        captions.append(vocab.word2idx["<unk>"])
                    len_token += 1  # 1 - 10
                    if len_token > 9:
                        break
                # padding
                while len_token <= 10:
                    captions.append(vocab.word2idx["<pad>"])
                    len_token += 1
                captions.append(vocab.word2idx["<end>"])
                # captions = torch.tensor(captions)
                yield features, captions


def train(features, captions):
    # get a fresh hidden layer
    # hidden = lstm.initHidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence
    # def forward(self, features, captions)
    predictions = lstm(features, captions)
    # compute loss (NLLH)
    loss = criterion(predictions[:-1], captions[1:len(captions)])
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
    ###############################

    # model training
    # lstm = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=0, num_layers=num_layers)
    #             # cat_embedding_size=32, n_cat=n_categories, ####### features
    #             # char_embedding_size=embed_size,
    #             # n_char=vocab_size,
    #             # output_size=vocab_size,
    # # training objective
    # criterion = nn.NLLLoss(reduction='sum')
    # # learning rate
    # learning_rate = 0.005
    # # optimizer
    # optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # # training parameters
    # n_iters = 50000
    # print_every = 5000
    # plot_every = 500
    # all_losses = []
    # total_loss = 0  # will be reset every 'plot_every' iterations
    #
    # # start = time.time()
    #
    # for i in range(1, n_iters + 1):
    #     loss = train(*getTrainingPair())
    #     total_loss += loss
    #
    #     if i % plot_every == 0:
    #         all_losses.append(total_loss / plot_every)
    #         total_loss = 0
    #         print(all_losses)
    ##############################

    for i in range(10):
        print(*getTrainingPair())



