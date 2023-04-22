import json
from collections import Counter
import nltk
from utils.build_vocab import Vocabulary
import pickle
from train.image_captioning.word_model import EncoderCNN
import torch
from torchvision import transforms
from os.path import isfile
from PIL import Image
from torch.autograd import Variable
import requests


embed_size = 256
hidden_size = 512
num_layers = 1

def getCocoPair():
    """
    get training item in desired input format (vectors of indices) from
    'coco_data/annotations_trainval2014/captions_train2014.json'
    :return:
    features: vector after pre-trained CNN
    captions: to be used in embeddings = self.embed(captions)
    """
    cnn = EncoderCNN(embed_size)
    cnn.eval()
    # self.encoder.load_state_dict(torch.load(self.encoder_path, map_location={'cuda:0': 'cpu'}))
    cnn.load_state_dict(torch.load('data/models/coco-encoder-5-3000.pkl', map_location={'cuda:0': 'cpu'}))
    if torch.cuda.is_available():
        cnn.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    json_data = json.loads(open('coco_data/annotations_trainval2014/captions_train2014.json', 'r').read())
    with open("coco_data/vocab_coco_4533.pkl", 'rb') as f:
        vocab = pickle.load(f)
    for each_img in json_data['annotations']:
        # {'image_id': 318556, 'caption': 'A very clean and well decorated empty bathroom'}
        image_id = each_img['image_id']  # 318556
        # http://images.cocodataset.org/train2014/COCO_train2014_000000318556.jpg
        file_url = 'http://images.cocodataset.org/train2014/COCO_train2014_000000' \
                   + str(image_id) + ".jpg"
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            # the picture exists
            im = Image.open(requests.get(file_url, stream=True).raw)
            # im.show()
            # resize
            im = im.resize([224, 224], Image.Resampling.LANCZOS)
            # transform (1, 3, 224, 224)
            im = transform(im).unsqueeze(0)  # tensor (1, 3, 224, 224)
            # boo = im1.size(dim=1) #== 3
            if im.size(dim=1) != 3:
                continue
            # feed into cnn
            # self.encoder(to_var(load_image(url, self.transform))); to_var() from utils/sample.py
            if torch.cuda.is_available():
                im = im.cuda()  # {Tensor: (1, 3, 224, 224)}
            features = cnn(Variable(im))  # cnn.forward(Variable(im1))

            # read captions, get index
            caption = each_img['caption']
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


if __name__ == '__main__':
    count = 0
    for i in getCocoPair():
        feat, cap = i
        count += 1
        # feature: tensor (1, 256); caption: tensor(12,)
        print(feat.shape)
        print(cap.shape)
        # last_time = time.time()
        if count > 0:
            break
    #########################
    # with open("coco_data/vocab_coco_4533.pkl", 'rb') as f:
    #     vocab = pickle.load(f)
    # print(vocab.word2idx['<start>'])
