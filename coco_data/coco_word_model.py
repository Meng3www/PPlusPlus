import torch.nn as nn
import torch
import time
from torch.utils.data import Dataset, DataLoader
import pickle


embed_size = 256
hidden_size = 512
num_layers = 1
vocab_size = 4533
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
    def forward(self, features, caption, hidden_prev):  ##
        """
        Decode image feature vectors and generates captions.
        """
        # print('captions: ', captions)  # device='cuda:0'
        embeddings = self.embed(caption)  # Tensor: (12, 256)
        # print('features.shape before cat: ', features.shape)  #
        # print('embeddings.shape before cat: ', embeddings.shape)  # torch.Size([12, 256])
        embeddings = torch.cat((features.repeat(caption.shape[0], 1), embeddings), 1).to(device)  # (12, 512)
        # print('embeddings.shape fed to lstm: ', embeddings.shape)  # torch.Size([12, 512])
        # output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim=1))
        # Defaults to zeros if (h_0, c_0) is not provided.
        # output, _ = self.lstm(embeddings)
        output, hidden = self.lstm(embeddings, hidden_prev)  ##
        # print('output shape: ', output.shape)  # torch.Size([12, 512])
        predictions = self.linear(output)
        # print('predictions shape: ', predictions.shape)  # torch.Size([12, 4987])
        # return torch.nn.functional.log_softmax(predictions, dim=1)
        return torch.nn.functional.log_softmax(predictions, dim=1), hidden  ##

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
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
            log_p = torch.log(max_probs)  # tensor (1, 1)
            # entropy = -log_p * max_probs  # tensor (1, 1)
            sampled_ids.append(caption)

        # print("SAMPLED IDS",sampled_ids.size())
        sampled_ids = torch.cat(sampled_ids, 0)  # (batch_size, 20)
        return sampled_ids.squeeze()

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


class TrainData(Dataset):
    def __init__(self, file_path):
        # data loading
        with open(file_path, 'rb') as f:
            self.data = torch.load(f, map_location=device)
        self.n_samples = len(self.data)

    def __getitem__(self, index):
        # return a pair of feature and captions, allowing indexing
        return self.data[index][0], self.data[index][1]# , self.data[index][2], self.data[index][3]

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
    shuffle = False
    dataset_0 = TrainData("coco_data/coco_train_0-10k.pkl")
    dataloader_0 = DataLoader(dataset=dataset_0, batch_size=1, num_workers=0, shuffle=shuffle)
    dataset_1 = TrainData("coco_data/coco_train_20-30k.pkl")
    dataloader_1 = DataLoader(dataset=dataset_1, batch_size=1, num_workers=0, shuffle=shuffle)
    dataset_2 = TrainData("coco_data/coco_train_31-39k.pkl")
    dataloader_2 = DataLoader(dataset=dataset_2, batch_size=1, num_workers=0, shuffle=shuffle)
    dataset_3 = TrainData("coco_data/coco_train_40-53k.pkl")
    dataloader_3 = DataLoader(dataset=dataset_3, batch_size=1, num_workers=0, shuffle=shuffle)

    # train without loading pre-trained model
    lstm = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
                      vocab_size=vocab_size, num_layers=num_layers).to(device)
    # lstm.load_state_dict(torch.load('07.pkl'))  ###
    # lstm.eval()  ###
    # lstm.train()  ###
    print('(pre-trained) model loaded, resume training')
    criterion = nn.NLLLoss(reduction='sum')
    # learning rate
    learning_rate = 0.01
    # optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # training parameters
    n_epoch = 2
    total_loss = 0
    save_every = 20  # save once 20 epochs are finished
    print("test learning rate: ", str(learning_rate))  ######
    start = time.time()
    print("epochs\tloss\t\t\ttotol time (min)")
    for epoch in range(1, n_epoch+1):
        for feature, captions in dataloader_0:
            # feature = feature.to(device)  # .to(device) redundant
            captions = captions.to(device)
            loss_s = train(feature[0], captions[0])
            total_loss += loss_s
            break  #############

        for feature, captions in dataloader_1:
            # feature = feature.to(device)  # .to(device) redundant
            captions = captions.to(device)
            loss_s = train(feature[0], captions[0])
            total_loss += loss_s
            break  #############

        for feature, captions in dataloader_2:
            # feature = feature.to(device)  # .to(device) redundant
            captions = captions.to(device)
            loss_s = train(feature[0], captions[0])
            total_loss += loss_s
            break  #############

        for feature, captions in dataloader_3:
            # feature = feature.to(device)  # .to(device) redundant
            captions = captions.to(device)
            loss_s = train(feature[0], captions[0])
            total_loss += loss_s
            break  #############

        print(epoch, "\t", round(total_loss, 2), "\t", round((time.time() - start)/60, 2))
        total_loss = 0

        if epoch % save_every == 0:  # checkout at each n epochs
            out_file_name = 'coco_word_decoder_' + str(epoch) + '_.pkl'
            torch.save(lstm.state_dict(), out_file_name)
            print('checkpoint saved as ', out_file_name)

