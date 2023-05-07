import torch.nn as nn
import torch
import time
from torch.utils.data import Dataset, DataLoader
import pickle
from utils.build_vocab import Vocabulary


embed_size = 256
hidden_size = 512
num_layers = 1
vocab_size = 4987
# https://www.youtube.com/watch?v=y2BaTt1fxJU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        # self.cat_embedding = nn.Embedding(n_cat, cat_embedding_size)  # (18, 32)
        self.embed = nn.Embedding(vocab_size, embed_size)
        # output, (h_n, c_n), input_size = cat_embedding_size+char_embedding_size (32+32=64)
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    # def forward(self, features, captions):
    def forward(self, features, captions, hidden_prev):  ##
        """
        Decode image feature vectors and generates captions.
        """
        # print('captions: ', captions)  # device='cuda:0'
        embeddings = self.embed(captions)  # Tensor: (12, 256)
        # print('features.shape before cat: ', features.shape)  #
        # print('embeddings.shape before cat: ', embeddings.shape)  # torch.Size([12, 256])
        embeddings = torch.cat((features.repeat(captions.shape[0], 1), embeddings), 1).to(device)  # (12, 512)
        # print('embeddings.shape fed to lstm: ', embeddings.shape)  # torch.Size([12, 512])
        # output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim=1))
        # Defaults to zeros if (h_0, c_0) is not provided.
        # output, _ = self.lstm(embeddings)  # _: (tensor (1, 512), tensor (1, 512))
        output, hidden = self.lstm(embeddings, hidden_prev)  ##
        # print('output shape: ', output.shape)  # torch.Size([12, 512])
        predictions = self.linear(output)
        # print('predictions shape: ', predictions.shape)  # torch.Size([12, 4987])
        # return torch.nn.functional.log_softmax(predictions, dim=1)
        return torch.nn.functional.log_softmax(predictions, dim=1), hidden  ##

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        log_p = 0
        softmax = nn.Softmax(dim=-1)
        # features: tensor (1, 256)
        for i in range(12):  # maximum sampling length
            if i == 0:
                caption = [1]  # vocab.word2idx['<start>']
                caption = torch.tensor(caption)  # tensor (1,)
                hidden = self.init_hidden()
            output, hidden = self.forward(features, caption, hidden)
            # hiddens, states = self.lstm(features, states)  # (batch_size, 1, hidden_size),
            # right: tensor (1, 4987) neg float
            # output = self.linear(output.squeeze(1))  # (batch_size, vocab_size)
            probs = softmax(output)  # tensor (1, 4987)
            max_probs, caption = torch.max(probs, dim=-1)  # tensor([4])
            log_p += torch.log(max_probs).item()  # tensor (1,) -> float
            # entropy = -log_p * max_probs  # tensor (1, 1)
            sampled_ids.append(caption)

        # print("SAMPLED IDS",sampled_ids.size())
        sampled_ids = torch.cat(sampled_ids, 0)  # (batch_size, 20)
        return sampled_ids.squeeze(), log_p

    def init_hidden(self):
        # (tensor(1, 512), tensor(1, 512))
        return (torch.zeros(1, hidden_size), torch.zeros(1, hidden_size))


def train(features, captions):
    # get a fresh hidden layer
    hidden = lstm.init_hidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence;  def forward(self, features, captions)
    # predictions = lstm(features, captions)
    predictions, _ = lstm(features, captions, hidden)  ##
    # compute loss (NLLH)
    loss = criterion(predictions[:-1], captions[1:len(captions)])
    # perform backward pass
    loss.backward()
    # perform optimization
    optimizer.step()
    # return prediction and loss
    return loss.item()


class ModelDataset(Dataset):
    def __init__(self, file_path):
        # data loading
        with open(file_path, 'rb') as f:
            self.data = torch.load(f, map_location=device)
        self.n_samples = len(self.data)

    def __getitem__(self, index):
        # return a pair of feature and captions, allowing indexing
        return self.data[index][0], self.data[index][1] , self.data[index][2], self.data[index][3]

    def __len__(self):
        return self.n_samples


def print_cap(cap_idx):
    caps = []
    with open("vg_data/vocab_small.pkl", 'rb') as f:
        vocab = pickle.load(f)  # a Vocabulary() from utils/build_vocab.py
    for idx in cap_idx:
        if idx == 2:
            break
        else:
            caps.append(vocab.idx2word[idx.item()])
    return " ".join(caps)


if __name__ == '__main__':

    vg_dataset = ModelDataset("vg_data/train_test_data/vg_test_0.pkl")
    dataloader = DataLoader(dataset=vg_dataset, batch_size=1, num_workers=0, shuffle=False)
    lstm = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
                      vocab_size=vocab_size, num_layers=num_layers).to(device)

    #################### train without loading pre-trained model
    # criterion = nn.NLLLoss(reduction='sum')
    # # learning rate
    # learning_rate = 0.0005
    # # optimizer
    # optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # # training parameters
    # n_epoch = 100
    # # print_every = 5
    # all_losses = []
    # total_loss = 0
    #
    # start = time.time()
    # print("epochs\tloss\t\t\ttime(s)")
    # for epoch in range(1, n_epoch+1):
    #     for feature, captions in dataloader:
    #         # feature = feature.to(device)  # tensor (1, 1, 256)
    #         captions = captions.to(device)  # tensor (1, 12)
    #         loss = train(feature[0], captions[0])
    #         total_loss += loss
    #
    # print(epoch, "\t", total_loss, "\t", time.time() - start)  # 5.8m/epoch
    # start = time.time()
    # total_loss = 0
    # # save: torch.save(model.state_dict(), PATH)
    # # torch.save(model, PATH)
    # torch.save(lstm.state_dict(), 'vg_word_decoder.pkl')

    ####################### sample
    lstm.load_state_dict(torch.load('vg_data/decoder/vg_word_decoder_105_04.pkl'))  ###
    # lstm.eval()  ###
    # lstm.train()  ###
    print('(pre-trained) model loaded')
    # for i, (feature, captions) in enumerate(dataloader):  # train_data
    for i, (feature, captions, phrase, url) in enumerate(dataloader):  # test_data
        feature = feature.to(device)  # tensor (1, 1, 256)
        captions = captions.to(device)  # tensor (1, 12)
        output, log_p = lstm.sample(feature[0])
        print("predicted: ", print_cap(output), "\t", str(log_p))
        print("target: ", phrase, url)
        if i > 5: break

