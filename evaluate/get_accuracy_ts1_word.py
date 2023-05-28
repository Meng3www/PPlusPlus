# This file is immplements an automatic evaluation to evaluate the success of S1 as compared to S0.
# It is similar to get_accuracy_ts1.py, but with some modifications for the word-level decoder.
# Also, due to the large vocab size of word-level decoder, this implementation only evaluates the
# captions generated via greedy, without beam search.

import torch
import numpy as np
import pickle
from build_test_data import build_img_path
from build_test_data import build_img_path
from utils.image_and_text_utils_word import vectorize_caption
from evaluate.decoder_word import DecoderRNN
from utils.sample import to_var,load_image_from_path
from train.image_captioning.char_model import EncoderCNN
from torchvision import transforms
from utils.build_vocab import Vocabulary

with open("coco_data/vocab_coco_4533.pkl", 'rb') as f:
	vocab = pickle.load(f)
	word2idx = vocab.word2idx

def get_caption_accuracy(encoder, decoder, transform, img_paths, s0_caption, s1_caption):

	s0_probs = list()
	s1_probs = list()

	# the caption strings are transformed to lists
	# in which each character is represented as an integer
	s0_captions = vectorize_caption(s0_caption)[0]
	s1_captions = vectorize_caption(s1_caption)[0]

	# iterate over the 10 images in this cluster
	for path in img_paths:
		# apply encoder to obtain image features
		features = encoder(to_var(load_image_from_path(path, transform), volatile=True))
		
		# cumulative log probability of s0 character
		s0_log_prob = 0

		# iterate over every character in s0 caption
		for i in range(len(s0_captions)-1):
			if i == 0:
				caption = [1]  # vocab.word2idx['<start>']
				hidden = decoder.init_hidden()
			else:
				caption = [word2idx[s0_captions[i]]]
			caption = torch.tensor(caption)  # tensor (1,)				
			output, hidden = decoder.forward(features, caption, hidden)

			# obtain the probability of the next char predicted by the model
			next_word = word2idx[s0_captions[i+1]]
			next_word_log_prob = output[0][next_word]
	
			# append the next char probability to the overall probability
			s0_log_prob += next_word_log_prob

		# append the probability of s0 caption being the caption of this image to the list
		s0_log_prob = s0_log_prob.detach().numpy().item()
		s0_probs.append(s0_log_prob)

	
		# do the same to obtain the probability of the s1 caption being the caption of this image
		s1_log_prob = 0
		for i in range(len(s1_captions)-1):
			if i == 0:
				caption = [1]  # vocab.word2idx['<start>']
				hidden = decoder.init_hidden()
			else:
				caption = [word2idx[s1_captions[i]]]
			caption = torch.tensor(caption)  # tensor (1,)				
			output, hidden = decoder.forward(features, caption, hidden)

			# obtain the probability of the next char predicted by the model
			next_word = word2idx[s1_captions[i+1]]
			next_word_log_prob = output[0][next_word]
	
			s1_log_prob += next_word_log_prob

		s1_log_prob = s1_log_prob.detach().numpy().item()
		s1_probs.append(s1_log_prob)

	# find for which image in the 10-image cluster are the s0 and s1 captions most probably for    
	s0_max = np.argmax(s0_probs)
	s1_max = np.argmax(s1_probs)

	print(s0_probs, s1_probs)

	# return whether the s0 and s1 captions most probably describe the target image among 10 images,
	# and whether the s1 caption is more probable for the target image than s0
	return s0_max==0, s1_max==0, s1_probs[0]>s0_probs[0]



if __name__ == "__main__":
	# initialize parameters
	embed_size = 256
	hidden_size = 512
	num_layers = 1
	vocab_size = 4987

	# load the decoder
	decoder = DecoderRNN(embed_size=embed_size, hidden_size=hidden_size,
				vocab_size=vocab_size, num_layers=num_layers)
	decoder.load_state_dict(torch.load('data/models/vg_word_decoder.pkl', map_location={'cuda:0': 'cpu'}))

	# load the encoder
	encoder = EncoderCNN(embed_size)
	encoder.eval()
	encoder.load_state_dict(torch.load("data/models/vg-encoder-5-3000.pkl", map_location={'cuda:0': 'cpu'}))

	# initialize transform for iamge features (obtained from author's code)
	transform = transforms.Compose([
			transforms.ToTensor(), 
			transforms.Normalize((0.485, 0.456, 0.406), 
								 (0.229, 0.224, 0.225))])

	# load the s0 and s1 captions generated by test images and coco char-level model
	ts1_captions = pickle.load(open("evaluate/ts1_captions_word",'rb'))

	# variables to record whether the greedy s0 and s1 captions succeeded in every cluster
	s0_accurate_greedy = []
	s1_accurate_greedy = []
	s1_higher_greedy = []

	# obtain all image paths of the test set 1 data set
	img_path = build_img_path("vg_data/ts1_img/")

	# iterate over every category in the dataset
	for category, value in img_path.items():
		print("------------------------------------------------")
		print("Test data in: " + category)

		# iterate over every cluster in the category
		for num, paths in value.items():
			print("Number: ", num)
			# obtain the captions generated for this cluster
			s0_caption_greedy = ts1_captions[category][num]["greedy"][0][0]
			s1_caption_greedy = ts1_captions[category][num]["greedy"][1][0]
			print("greedy")
			# test the s0 and s1 captions by greedy
			greedy_accurate = get_caption_accuracy(encoder, decoder, transform, paths, s0_caption_greedy, s1_caption_greedy)
			s0_accurate_greedy.append(greedy_accurate[0])
			s1_accurate_greedy.append(greedy_accurate[1])
			s1_higher_greedy.append(greedy_accurate[2])

	# print the results
	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	print("---------- greedy ------------")
	print("S0 accuracy greedy: ", np.count_nonzero(s0_accurate_greedy))
	print("S1 accuracy greedy: ", np.count_nonzero(s1_accurate_greedy))
	print("S1 higher greedy: ", np.count_nonzero(s1_higher_greedy))

	# save the results in a file
	pickle.dump((greedy_accurate), open('evaluate/ts1_accuracy','wb'))
	