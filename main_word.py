# this code will generate a literal caption and a pragmatic caption (referring expression) for the first of the urls provided in the context of the rest

import matplotlib
matplotlib.use('Agg')
from utils.numpy_functions import uniform_vector, make_initial_prior
from recursion_schemes.recursion_schemes_word import ana_greedy,ana_beam
from bayesian_agents.joint_rsa_word import RSA
from utils.build_vocab import Vocabulary

urls = [
    "vg_data/ts1_img/man/1_3082",
	"vg_data/ts1_img/man/51_4931359"
	]

# code is written to be able to jointly infer speaker's rationality and neural model, but for simplicity, let's assume these are fixed
# the rationality of the S1
# rat = 100, beam s1 produces non words
# rat = 5, coco + beam s1 produces sentences ending with '$'
# rat = 5, coco + beam s1 say things do not appear on the picture still, but '$'s are reduced
rat = [5.0]
# the neural model: captions trained on MSCOCO ("coco") are more verbose than VisualGenome ("vg")
model = ["coco"]
number_of_images = len(urls)
# the model starts of assuming it's equally likely any image is the intended referent
initial_image_prior=uniform_vector(number_of_images)
initial_rationality_prior=uniform_vector(1)
initial_speaker_prior=uniform_vector(1)
initial_world_prior = make_initial_prior(initial_image_prior,initial_rationality_prior,initial_speaker_prior)

# make a character level speaker, using torch model (instead of tensorflow model)
speaker_model = RSA(seg_type="word",tf=False)
speaker_model.initialize_speakers(model)
# set the possible images and rationalities
#speaker_model.speaker_prior.set_features(images=urls,tf=False,rationalities=rat)  # list: 2, 2 tensors of shape (1, 256)
speaker_model.initial_speakers[0].set_features(images=urls,tf=False,rationalities=rat)  # list: 2, 2 tensors of shape (1, 256)
# generate a sentence by unfolding stepwise, from the speaker: greedy unrolling used here, not beam search: much better to use beam search generally
literal_caption = ana_greedy(
# literal_caption = ana_beam(
	speaker_model,
	target=0,
	depth=0,
	speaker_rationality=0,
	speaker=0,
	start_from=list(""),
	initial_world_prior=initial_world_prior
#	, beam_width=2
)

pragmatic_caption = ana_greedy(
# pragmatic_caption = ana_beam(
	speaker_model,
	target=0,
	depth=1,
	speaker_rationality=0,
	speaker=0,
	start_from=list(""),
	initial_world_prior=initial_world_prior
#	, beam_width=2
)


print("Literal caption:\n",literal_caption)
print("Pragmatic caption:\n",pragmatic_caption)
