import torch.nn as nn
import torch
import time
from torch.utils.data import Dataset, DataLoader
import pickle

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

    def forward(self, features, captions):
        """
        Decode image feature vectors and generates captions.
        """
        # print('captions: ', captions)  # device='cuda:0'
        embeddings = self.embed(captions)  # Tensor: (12, 256)
        # print('features.shape before cat: ', features.shape)  #
        # print('embeddings.shape before cat: ', embeddings.shape)  # torch.Size([12, 256])
        embeddings = torch.cat((features.repeat(12, 1), embeddings), 1).to(device)  # (12, 512)
        # print('embeddings.shape fed to lstm: ', embeddings.shape)  # torch.Size([12, 512])
        # output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim=1))
        # Defaults to zeros if (h_0, c_0) is not provided.
        output, _ = self.lstm(embeddings)
        # print('output shape: ', output.shape)  # torch.Size([12, 512])
        predictions = self.linear(output)
        # print('predictions shape: ', predictions.shape)  # torch.Size([12, 4987])
        return torch.nn.functional.log_softmax(predictions, dim=1)

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        with open("data/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)  # a Vocabulary() from utils/build_vocab.py
        for i in range(20):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size),

            # print(hiddens.size())
            # print(states[0].size(),states[1].size())

            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]

            # print("stuff",type(predicted.data),predicted.data)
            # print(vocab.idx2word[1])
            # print("\nNNASDFKLASDJF\n\n",vocab.idx2word[predicted.data.cpu().numpy()[0]])

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)

        # print("SAMPLED IDS",sampled_ids.size())
        sampled_ids = torch.cat(sampled_ids, 0)  # (batch_size, 20)
        return sampled_ids.squeeze()

    def init_hidden(self):
        return torch.zeros(1, hidden_size)


def train(features, captions):
    # get a fresh hidden layer
    # hidden = lstm.initHidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence;  def forward(self, features, captions)
    predictions = lstm(features, captions)
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
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return self.n_samples


if __name__ == '__main__':

    vg_dataset = ModelDataset("vg_feat_cap_0.pkl")
    dataloader = DataLoader(dataset=vg_dataset, batch_size=1, num_workers=0, shuffle=True)

    # train without loading pre-trained model
    lstm = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
                      vocab_size=vocab_size, num_layers=num_layers).to(device)
    lstm.load_state_dict(torch.load('vg_word_decoder_100_07.pkl'))  ###
    # lstm.eval()  ###
    # lstm.train()  ###
    print('(pre-trained) model loaded, resume training')
    criterion = nn.NLLLoss(reduction='sum')
    # learning rate
    learning_rate = 0.00004
    # optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # training parameters
    n_epoch = 10
    # print_every = 1
    all_losses = []
    total_loss = 0

    start = time.time()
    print("epochs\tloss\t\t\ttotol time (min)")
    for epoch in range(101, n_epoch+101):
        for feature, captions in dataloader:
        # feature = feature.to(device)  # .to(device) redundant
            captions = captions.to(device)
            loss = train(feature[0], captions[0])
            total_loss += loss

        print(epoch, "\t", total_loss, "\t", (time.time() - start)/60)
        # start = time.time()
        total_loss = 0

        if epoch % 5 == 0: # checkout at each n epochs
            out_file_name = 'vg_word_decoder_' + str(epoch) + '_04.pkl'
            torch.save(lstm.state_dict(), out_file_name)
            print('checkpoint saved as ', out_file_name)

