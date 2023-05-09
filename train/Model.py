import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from utils.build_vocab import Vocabulary
from train.word_model import EncoderCNN, DecoderRNN
from PIL import Image
import torch
from utils.config import *
from utils.numpy_functions import softmax


class Model:

	def __init__(self,path,dictionaries):
		
		self.seg2idx,self.idx2seg=dictionaries
		self.path=path
		self.vocab_path='data/vocab/' + path + '_vocab_small.pkl'  # data/vocab/vg_vocab_small.pkl
		self.encoder_path=TRAINED_MODEL_PATH+path+"-encoder-5-3000.pkl"
		# self.decoder_path=TRAINED_MODEL_PATH+path+"-decoder-5-3000.pkl"  # char
		# data/models/vg_word_decoder.pkl
		self.decoder_path = TRAINED_MODEL_PATH + path + "_word_decoder.pkl"  # word

		#todo: change
		embed_size=256
		hidden_size=512
		num_layers=1

		output_size = len(self.seg2idx)

		transform = transforms.Compose([
			transforms.ToTensor(), 
			transforms.Normalize((0.485, 0.456, 0.406), 
								 (0.229, 0.224, 0.225))])
		
		self.transform = transform
		# Load vocabulary wrapper


		# Build Models
		self.encoder = EncoderCNN(embed_size)
		self.encoder.eval()  # evaluation mode (BN uses moving mean/variance)
		# eval() turns off some specific layers/parts of the model that behave differently during training 
		# and inference (evaluating), e.g. Dropouts Layers, BatchNorm Layers
		# the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() 
		# to turn off gradients computation
		self.decoder = DecoderRNN(embed_size, hidden_size, output_size, num_layers)

		# Load the trained model parameters
		# encoder: linear = {Linear} Linear(in_feature=2048, out_feature=256, bias=True)
		self.encoder.load_state_dict(torch.load(self.encoder_path, map_location={'cuda:0': 'cpu'}))
		# DecoderRNN(
		#   (embed): Embedding(30, 256)
		#   (lstm): LSTM(256, 512, batch_first=True)
		#   (linear): Linear(in_features=512, out_features=30, bias=True)
		# )
		self.decoder.load_state_dict(torch.load(self.decoder_path, map_location={'cuda:0': 'cpu'}))
		# self.embed = nn.Embedding(output_size, embed_size)  # embedding from the decoder
		if torch.cuda.is_available():
			self.encoder.cuda()
			self.decoder.cuda()

	def forward(self,world,state):
		# state {context_sentence: [], world_priors: ndarray: (61, 2, 1, 1) [[[[-0.69314718]],,  [[-0.69314718]]],,, ...}
		# world <World image:0 rationality:0 speaker:0>
		# world.target: 0, inputs: tensor(1, 1, 256)
		feature = self.features[world.target].unsqueeze(1)
		states = None

		for seg in state.context_sentence:  # '<start>'	[1] # maximum sampling length
			caption = [self.seg2idx[seg]]  # list [1]
			caption = torch.tensor(caption)  # tensor (1,)
			embeddings = self.decoder.embed(caption).unsqueeze(1)  # tensor (1, 1, 256)

			inputs = torch.cat((feature, embeddings), -1)  # (1, 1, 512)
			# states (tensor (1, 1, 512), tensor (1, 1, 512))
			hiddens, states = self.decoder.lstm(inputs, states)	 # hidden: tensor (1, 1, 512)	  # (batch_size, 1, 512),
			# tensor (1, 4987)
			outputs = self.decoder.linear(hiddens.squeeze(1)) 
			# tensor (1, ) index
			predicted = outputs.max(1)[1]  # tensor (1,) index ##############################
			word_pred = self.idx2seg[predicted.item()]  ##
			predicted[0] = self.seg2idx[seg]
			# tensor (1, 256)
			inputs = self.decoder.embed(predicted)
			# tensor (1, 1, 256)
			inputs = inputs.unsqueeze(1)  # (1, 1, 256)
			inputs = torch.cat((feature, inputs), -1)  # (1, 1, 512)
		# inputs: tensor (1, 1, 512), states: (tensor (1, 1, 512), tensor (1, 1, 512))
		hiddens, states = self.decoder.lstm(inputs, states)		  # (batch_size, 1, 512),
		outputs = self.decoder.linear(hiddens.squeeze(1))   # tensor (1, 4987) float
		output_array = outputs.squeeze(0).data.cpu().numpy()  # ndarray (4987,) float

		log_softmax_array = np.log(softmax(output_array))  # ndarray (4987,) float
		return log_softmax_array

	def set_features(self,images,rationalities,tf):

		self.number_of_images = len(images)
		self.number_of_rationalities = len(rationalities)  # 1
		self.rationality_support=rationalities  # [1.0]

		if tf:
			pass

		else:
			from utils.sample import to_var,load_image
			# the list of urls from main. list: len(urls), tensors of shape (1, 256)
			self.features = [self.encoder(to_var(load_image(url, self.transform), volatile=True)) for url in images]
			# self.default_image = self.encoder(to_var(load_image_from_path("data/default.jpg", self.transform), volatile=True))

			# self.speakers = [Model(path) for path in paths]

			# imgs = [load_image(url) for url in urls]
			# self.images=[]

			# for img in imgs:
			# 	img_array = np.expand_dims(image.img_to_array(img),0)
			# 	img_rep = resnet(img_rep_layer).predict(img_array)
			# 	self.images.append(img_rep)

			# self.images = np.asarray(self.images)


if __name__ == '__main__':
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406),
							 (0.229, 0.224, 0.225))])
	print(transform)
	image = Image.open('../data/google_images/test.jpg')
	print(image)
	print(transform(image).shape)
	print(transform(image).unsqueeze(0).shape)

# image = transform(image).unsqueeze(0)



