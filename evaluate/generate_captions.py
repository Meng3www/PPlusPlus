# This code generates a literal caption (s0) and a pragmatic caption (s1) 
# for the first of the image paths provided in the context of the rest of the image paths.
# It uses the images from test set 1, which contains 10 objects x 10 clusters x 10 images,
# and deploys the "coco" encoder and decoder for caption generation.
# The captions are used for evaluation of the success of s1 in comparison to s0

import matplotlib
matplotlib.use('Agg')
from utils.numpy_functions import uniform_vector, make_initial_prior
from recursion_schemes.recursion_schemes import ana_greedy,ana_beam
from bayesian_agents.joint_rsa import RSA
from build_test_data import build_img_path
import pickle

def generate_captions(image_paths, s1_rationality=[5.0], seg_type="char", model=["coco"]):
	'''
    returns a dictionary of captions, with "greedy" corresponds to the s0 and s1 captions and probabilities
	generated by greedy unrolling, and "beam" corresponds to those by beam search, e.g.:
    {
		'greedy': [
			('^a woman is standing next to a woman with a cat on his lap&$', -21.109453), 
			('^the two women are playing a video game&$', -9.432134133490036)
			], 
		'beam': [
			('^a group of people standing next to each other&$', -7.101910561763475), 
			('^three people playing video games together& $', -4.623391648254756)
			]
	}

	parameters:
	image_paths: a list of image paths, with the first image being the target
	s1_rationality: s1 rationality, default is 5.0
	seg_type: char or word, default is char
	model: coco or vg, the decoder and encoder to use
    '''
	# the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")
	number_of_images = len(image_paths)	# 10

	# the model starts of assuming it's equally likely any image is the intended referent
	initial_image_prior=uniform_vector(number_of_images)  # array([0.1, 0.1, 0.1, ..., 0.1]) 
	initial_rationality_prior=uniform_vector(1)				# array([1.])
	initial_speaker_prior=uniform_vector(1)					# array([1.])
	initial_world_prior = make_initial_prior(initial_image_prior, initial_rationality_prior, initial_speaker_prior)
	# [0:10]: [array([[-2.30]]), array([[-2.30]]), ..., array([[-2.30]])]

	# make a character level speaker, using torch model (instead of tensorflow model)
	speaker_model = RSA(seg_type, tf=False)	# idx2seg, seg2idx <defaultdict, len=30>
	speaker_model.initialize_speakers(model)
	# embed_size: 256, hidden_size:512, output_size: 30, 
	# speaker_model.encoder(): self.resnet, self.linear, self.bn, self.linear.weight, self.linear.bias 
	# speaker_model.decoder(): self.embed: Embedding(30, 256), self.lstm: LSTM(256, 512, batch_first=True), 
	# 	self.linear: Linear(in_features=512, out_features=30, bias=True), self.embed.weight, self.linear.weight, self.linear.bias
	# load encoder, decoder pickle files

	# set the possible images and rationalities
	speaker_model.initial_speakers[0].set_features(images=image_paths,tf=False,rationalities=s1_rationality)  # list: 10, 10 tensors of shape (1, 256)

	# initialize an empty dictionary for captions
	captions = dict()
	captions["greedy"] = list()
	captions["beam"] = list()

	# generate a sentence by unfolding stepwise using greedy unrolling
	literal_caption_greedy = ana_greedy(
		speaker_model,
		target=0,
		depth=0,
		speaker_rationality=0,
		speaker=0,
		start_from=list(""),
		initial_world_prior=initial_world_prior
	)		
	pragmatic_caption_greedy = ana_greedy(
		speaker_model,
		target=0,
		depth=1,
		speaker_rationality=0,
		speaker=0,
		start_from=list(""),
		initial_world_prior=initial_world_prior
	)

	# generate a sentence by unfolding stepwise with beam search
	literal_caption_beam = ana_beam(
		speaker_model,
		target=0,
		depth=0,
		speaker_rationality=0,
		speaker=0,
		start_from=list(""),
		initial_world_prior=initial_world_prior,
	 	beam_width=10
	)
	pragmatic_caption_beam = ana_beam(
		speaker_model,
		target=0,
		depth=1,
		speaker_rationality=0,
		speaker=0,
		start_from=list(""),
		initial_world_prior=initial_world_prior,
	 	beam_width=10
	)

	# append the s0, s1 captions with greedy and beam to the dictionary
	captions["greedy"].append(literal_caption_greedy[0])
	captions["greedy"].append(pragmatic_caption_greedy[0])
	captions["beam"].append(literal_caption_beam[0])
	captions["beam"].append(pragmatic_caption_beam[0])

	return captions


if __name__ == "__main__":
	# prepare image paths of the test set 1 data set
	img_path = build_img_path("vg_data/ts1_img/")

	# initialize emtpy dictionary for captions
	ts1_captions = dict()

	# iterate over every category in the dataset
	for category, value in img_path.items():
		print("GENERATING CAPTIONS FOR: " + category)
		ts1_captions[category] = dict()

		# iterate over every cluster in the category
		for num, paths in value.items():
			print(num)
			# pass the paths in this cluster to generate captions
			# and save the result in the dictionary
			ts1_captions[category][num] = generate_captions(paths)

	# save the generated captions in a pickle file
	pickle.dump(ts1_captions, open('evaluate/ts1_captions','wb'))
		

