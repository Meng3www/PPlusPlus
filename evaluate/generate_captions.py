# this code will generate a literal caption and a pragmatic caption (referring expression) for the first of the urls provided in the context of the rest

import matplotlib
matplotlib.use('Agg')
from utils.numpy_functions import uniform_vector, make_initial_prior
from recursion_schemes.recursion_schemes import ana_greedy,ana_beam
from bayesian_agents.joint_rsa import RSA

def generate_captions(image_paths, s1_rationality=[5.0], greedy=True, seg_type="char", model=["coco"]):

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

	# generate a sentence by unfolding stepwise, from the speaker: greedy unrolling used here, not beam search: much better to use beam search generally
	if greedy:
		literal_caption = ana_greedy(
			speaker_model,
			target=0,
			depth=0,
			speaker_rationality=0,
			speaker=0,
			start_from=list(""),
			initial_world_prior=initial_world_prior
		)
		pragmatic_caption = ana_greedy(
			speaker_model,
			target=0,
			depth=1,
			speaker_rationality=0,
			speaker=0,
			start_from=list(""),
			initial_world_prior=initial_world_prior
		)
	else:
		literal_caption = ana_beam(
			speaker_model,
			target=0,
			depth=0,
			speaker_rationality=0,
			speaker=0,
			start_from=list(""),
			initial_world_prior=initial_world_prior,
	 		beam_width=10
		)
		pragmatic_caption = ana_beam(
			speaker_model,
			target=0,
			depth=1,
			speaker_rationality=0,
			speaker=0,
			start_from=list(""),
			initial_world_prior=initial_world_prior,
	 		beam_width=10
		)

	print("Literal caption:\n",literal_caption)
	print("Pragmatic caption:\n",pragmatic_caption)
