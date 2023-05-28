import json
import nltk
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
# from utils.build_vocab import Vocabulary
import pickle
from os.path import isfile
from PIL import Image
from torchvision import transforms


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        resnet = models.resnet152(weights='DEFAULT')
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # Linear(in_features=2048, out_features=256, bias=True)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        # BatchNorm1d(256, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
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
        features = self.bn(self.linear(features))  # {Tensor: (1, 256)}
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # Embedding(4987, 256)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # LSTM(512, 512, batch_first=True)
        self.lstm = nn.LSTM(embed_size*2, hidden_size, num_layers, batch_first=True)
        # Linear(in_features=512, out_features=4987, bias=True)
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
        #captions = captions.clone().detach().long()  # Tensor: (12,)
        embeddings = self.embed(captions)  # Tensor: (12, 256)
        # print(features.shape)  #
        # print(features.unsqueeze(1).shape)  #
        # print(embeddings.shape)  #
        embeddings = torch.cat((features.repeat(12, 1), embeddings), 1)  # (12, 512)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # Defaults to zeros if (h_0, c_0) is not provided.
        # (12, 512)
        output, _ = self.lstm(embeddings)
        # get LSTM outputs
        # lstm_output, (h, c) = self.lstm(x, hidden)

        predictions = self.linear(output)  # (12, 4987)
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
        return torch.zeros(1, hidden_size)


embed_size=256
hidden_size=512
num_layers=1
vocab_size=4987


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
    cnn.load_state_dict(torch.load('data/models/vg-encoder-5-3000.pkl', map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        cnn.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    json_data = json.loads(open('vg_data/region_descriptions.json', 'r').read())
    with open("vg_data/vocab_small.pkl", 'rb') as f:
        vocab = pickle.load(f)  # a Vocabulary() from utils/build_vocab.py

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
                # im1.show()
                # resize
                im1 = im1.resize([224, 224], Image.Resampling.LANCZOS)
                # transform (1, 3, 224, 224)
                im1 = transform(im1).unsqueeze(0)  # tensor (1, 3, 224, 224)
                # boo = im1.size(dim=1) #== 3
                if im1.size(dim=1) != 3:
                    continue
                # feed into cnn
                # self.encoder(to_var(load_image(url, self.transform)))
                # to_var() from utils/sample.py
                if torch.cuda.is_available():
                    im1 = im1.cuda()  # {Tensor: (1, 3, 224, 224)}
                features = cnn(Variable(im1))  # cnn.forward(Variable(im1))

                # read captions, get index
                caption = each_region['phrase']
                tokens = nltk.tokenize.word_tokenize(caption.lower())
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
                captions.append(vocab.word2idx["<end>"])
                # padding
                while len_token < 10:
                    captions.append(vocab.word2idx["<pad>"])
                    len_token += 1
                captions = torch.tensor(captions)
                yield features, captions


def train(features, captions):
    # get a fresh hidden layer
    # hidden = lstm.init_hidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence
    # for i in range(captions.size(0)-1):
    #     caption = torch.tensor([captions[i]])
    #     # def forward(self, features, captions)
    #     output, hidden = lstm(features, caption, hidden)
    #     # compute loss (NLLH)
    #     l = criterion(output, captions[i+1])
    #     loss += l
    # feature: tensor (1, 256); caption: tensor(12,)
    predictions = lstm(features, captions)  # predictions {tensor: (name_length, 60)}
    # compute loss (NLLH) loss {tensorL ()} tensor(45.3363, grad_fn=<NllLossBackward0>)
    # print(predictions)
    # print(predictions.shape)
    loss = criterion(predictions[:-1], captions[1:len(captions)])
    # print(loss)
    # perform backward pass
    loss.backward()
    # perform optimization
    optimizer.step()
    # return prediction and loss
    return loss.item()


if __name__ == '__main__':
    import time

    # temp_dict = ["a", "list", "of", "words"]
    # with open("test/test.pkl", 'wb') as f:
    #     pickle.dump(temp_dict, f)
    # with open("test/test.pkl", 'rb') as f:
    #     vocab = pickle.load(f)
    # print(vocab)

    ###############################

    # last_time = begin = time.time()
    count = 0
    for i in getTrainingPair():
        feat, cap = i
        count += 1
        # feature: tensor (1, 256); caption: tensor(12,)
        print(feat.shape)
        print(cap.shape)
            #last_time = time.time()
        if count > 0:
            break
    # print('{0:30} {1}'.format('finished in', time.time() - begin))

    ###################################
    # model training
    lstm = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size, vocab_size=vocab_size, num_layers=num_layers)
    # training objective
    criterion = nn.NLLLoss(reduction='sum')
    # learning rate
    learning_rate = 0.005
    # optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # training parameters
    n_iters = 50000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0  # will be reset every 'plot_every' iterations

    # start = time.time()
    # cap = [i for i in range(12)]
    # cap = torch.as_tensor(cap)
    # feat = torch.zeros(1, 256)
    for i in range(1, 2):  # n_iters + 1):
        loss = train(feat, cap)
        total_loss += loss

        if i % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0
            print(all_losses)
    ##############################
    # RuntimeError: Tensors must have same number of dimensions: got 3 and 2