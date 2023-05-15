import numpy as np
from utils.image_and_text_utils_word import max_sentence_length,vectorize_caption

class RSA_State:

	def __init__(
		self,
		initial_world_prior,
		listener_rationality=1.0
		):
		# should deprecate these two above

		# vectorize_caption(sentence)-> caption_in (# ndarray: (61, ), [1 2 0 ... 0]), caption_out (ndarray: (61, 30))
		# context_sentence: ndarray: (1, 61, 1) [[[1],  [2],  [0], ... , [0]]]
		self.context_sentence=np.expand_dims(np.expand_dims(vectorize_caption("")[0],0),-1)
		
		# priors for the various dimensions of the world
		#  keep track of the dimensions of the prior
		self.dim = {"image":0,"rationality":1,"speaker":2}

		# the priors at t>0 only matter if we aren't updating the prior at each step
		# initial_world_prior: ndarray (2, 1, 1) [[[-0.69314718]],, [[-0.69314718]]]
		# self.world_priors: ndarray: (61, 2, 1, 1) [[[[-0.69314718]],,  [[-0.69314718]]],,, ...
		self.world_priors=np.asarray([initial_world_prior for x in range(max_sentence_length+1)])
		
		self.listener_rationality=listener_rationality  # 1.0

		# this is a bit confusing, isn't it
		self.timestep=1


	def __hash__(self):
		return hash((self.timestep,self.listener_rationality,tuple(self.context_sentence)))

	# def __eq__(self,other):
	# 	return self.target==other.target and self.speaker==other.speaker and self.rationality==other.rationality


