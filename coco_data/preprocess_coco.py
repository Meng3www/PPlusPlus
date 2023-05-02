import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torchvision import transforms
import json
import pickle
import requests
from PIL import Image
import nltk
import time
# nltk.download('punkt')


embed_size=256
hidden_size=512
num_layers=1
vocab_size=4533
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        # BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        resnet = models.resnet152(weights='DEFAULT')
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
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


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def getCocoData():
    """
    get training item in desired input format (vectors of indices) from
    'coco_data/annotations_trainval2014/captions_train2014.json'
    :return:
    features: vector after pre-trained CNN
    captions: to be used in embeddings = self.embed(captions)
    phrase: caption string
    file_url: url containing the image
    """
    cnn = EncoderCNN(embed_size)
    cnn.eval()
    # self.encoder.load_state_dict(torch.load(self.encoder_path, map_location={'cuda:0': 'cpu'}))
    cnn.load_state_dict(torch.load('data/models/coco-encoder-5-3000.pkl', map_location=device))
    if torch.cuda.is_available():
        cnn.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    json_data = json.loads(open('coco_data/trainval2014/captions_train2014.json', 'r').read())
    with open("coco_data/vocab_coco_4533.pkl", 'rb') as f:
        vocab = pickle.load(f)
    for each_img in json_data['annotations']:
        # {'image_id': 318556, 'caption': 'A very clean and well decorated empty bathroom'}
        image_id = each_img['image_id']  # 318556
        # http://images.cocodataset.org/train2014/COCO_train2014_000000318556.jpg
        file_url = 'http://images.cocodataset.org/train2014/COCO_train2014_000000' + str(image_id) + ".jpg"
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            # the picture exists
            im = Image.open(requests.get(file_url, stream=True).raw)
            if im.mode != 'RGB': continue
            # im.show()
            # resize
            im = im.resize([224, 224])
            # transform (1, 3, 224, 224)
            im = transform(im).unsqueeze(0)  # tensor (1, 3, 224, 224)
            # feed into cnn
            # self.encoder(to_var(load_image(url, self.transform))); to_var() from utils/sample.py
            if torch.cuda.is_available():
                im = im.cuda()  # {Tensor: (1, 3, 224, 224)}
            features = cnn(Variable(im))  # cnn.forward(Variable(im1))

            # read captions, get index
            phrase = each_img['caption']
            tokens = nltk.tokenize.word_tokenize(phrase.lower())
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
            yield features, captions, phrase, file_url


if __name__ == '__main__':
    # print()
    # for feature, captions, phrase, file_url in getCocoData():
    #     print(feature.shape)
    #     print(captions)
    #     print(phrase)
    #     print(file_url)
    #     break
    #########################
    # 40k pairs for training
    batch_temp = []  # batch container
    counter = 0
    start = time.time()
    file_index = 0
    for feature, captions, phrase, file_url in getCocoData():
        counter += 1
        if counter < 40000:  # save every 40k in one pkl
            batch_temp.append((feature, captions))
            if counter % 4000 == 0:  # time tracking
                print(counter, "\t", str(round((time.time() - start)/60, 1)))
        elif counter == 40000:
            batch_temp.append((feature, captions))
            filename = "coco_train_40k.pkl"
            with open(filename, 'wb') as fp:
                torch.save(batch_temp, fp)
            batch_temp = []  # reset the container
            print(str(counter), "\t", str(round((time.time() - start)/60, 1)), "\t", filename)
        else:  # > 40k
            batch_temp.append((feature, captions, phrase, file_url))
            if counter % 4000 == 0:
                filename = "coco_test4k_" + str(file_index) + ".pkl"
                file_index += 1
                with open(filename, 'wb') as fp:
                    torch.save(batch_temp, fp)
                batch_temp = []  # reset the container
                print(str(counter), "\t", str(round((time.time() - start)/60, 1)), "\t", filename)
        ##############
        # if counter == 40:
        #     print(time.time() - start)
        #     break
        ##############

    # if len(batch_temp) != 0:  # save the last bits that is fewer than 40k
    #     filename = 'coco_test' + str(len(batch_temp)) + ".pkl"
    #     with open(filename, 'wb') as fp:
    #         torch.save(batch_temp, fp)
    #     print("end after" + "\t" + str(round((time.time() - start)/60, 1)) + "\t\t" + filename)
    # print(str(counter), "\tpairs in total")

    ###################

