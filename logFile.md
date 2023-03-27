### 30.01.2023
decide which paper to reproduce 

| [Andreas & Klein (2016)](/Relevant%20Papers/Andreas%20%26%20Klein%202016%20Reasoning%20about%20Pragmatics.pdf) | [Cohn-Gordon et al (2018)](/Relevant%20Papers/Cohn-Gordon%20et%20al%202018%20Character-Level%20Inference.pdf) |
| ------------- | ------------- |
| insufficient information on encoding image | [code is provided](https://github.com/reubenharry/Recurrent-RSA) |
| human evaluation using Amazon Mechanical Turk | automatic evaluation: production and evaluation models trained on separate sets (?) |

can we get away with not having human evaluation for Andreas & Klein (2016): No considering fig 5n etc. <br>
acting as human rater means we two have to rate all the results $:)$

What to do with Cohn-Gordon et al:
- try to understand their paper and code
- reproduce
- control experiment with beam search for both, and greedy for both char and word


----------------------------------
### 31.01.2023
Notes on Cohn-Gordon et al.:  
Training:
- S0: use a character-level LSTM defining a distribution over characters P(u|pc, image)
    - pc: partial caption, string of characters constituting the caption so far
    - u: the next character of the character
    - pros: character-level U is much smaller than word-level
- L0: takes a partial caption and a new character, returns a distribution over images
- S1: takes a target image, performs inference over set of possible characters

Evaluation: 
- define a listener L_eval that use Bayes' rule to obtain from S0 the posterior probability of each image w given a full caption u
    - split the training data in half, 1 for training the S0 used in caption generation model S1, 1 for training the S0 used in the caption evaluation model L_eval
    - the caption succeeds as a referring expression if the target has more probability mass under the distribution L eval (image|caption) than any distractor.

Dataset: 
- Visual Genome dataset: provides captions for regions within images
- MSCOCO: captions for whole images
- 2 test sets
    - TS1: 100 cluster of images, 10 for each of the 10 most common objects in Visual Genome
    - TS2: 100 clusters of 10, regions in Visual Genome images whose ground truth captions have high word overlap (similar)
- neural image captioning system: a CNN-RNN architecture 4 adapted to use a character-based LSTM for the language
model
- use a beam search with width 10
- rationality parameter alpha=5.0 for S1

Results:
- compare the performance of character-level model to word-level model
    - word-level model is incremental, use a word-level LSTM, evaluated with an L_eval model that also operates on the word level
- Vriants of the model:
    - a variant of S1: has a prior over utterances determined by an LSTM language model trained on the full set
of captions, 67.2%
    - standard S1: with unrolling such that the L 0 prior is drawn uniformly at each timestep rather than determined by the L0 posterior at the previous step, 67.4%

Questions for tomorrow:
- what is the w' and u' in the formula?
    they are placeholders
- is files with ".pkl" trained encoder/decoder or data set?
    pickle is not important, any file that can hold data will do
    vocal.pkl: replacement with pytorch function that extracts vocab. more info on Google


------------------------------
### 31.01.2023
booked the appointment slot for 1st course project feedback
- 01.02.2023 9:50-10:10

Comment:  
Team members: Fanyi Meng, Jia Sheng

Project information: This project aims to reproduce the key results of the paper "Cohn-Gordon et al. (2018), Pragmatically Informative Image Captioning with Character-Level Inference" and critically access its evaluation approaches with beam search and greedy sampling for the character- and word-level incremental predictions.

Aspired submission date: April 15th, 2023

Number of ECTS points: 6

Some thoughts:

- try to understand their paper and code
- reproduce the results
- control experiment with beam search for both, and greedy for both char and word


------------------------------
### 01.02.2023
**priority** code, data set up and running

**to-dos**
- make the code our own (with modification and documentation)
- run some examples before training, eg. for word base beam search
- t-test is sufficient 
- performance metrics


------------------------------
### 21.03.2023
error "_pickle.UnpicklingError: invalid load key, 'v'." when running "main.py", line 39
- similar problem: https://stackoverflow.com/questions/56633136/receiving-unpicklingerror-invalid-load-key-v-when-trying-to-run-truecase
- solution:
    1. install git lfs: https://git-lfs.com/
    2. go to directory Recurrent-RSA and do "git lfs pull"

error "PIL.UnidentifiedImageError", utils.sample.py
- cause: `out_file` having the same file name as `image`, when `out_file` does not open properly and replaced the original image file
- solution: modify the code for naming `out_file` ***whether it's correct will be seen***
- **update on 23 March** the code for naming `out_file` is changed to the original, since the error comes from `Response [403]` and can be solved by changing url for pictures

error "Module 'scipy.misc' has no attribute 'logsumexp'" bayesian_agents.joint_rsa.py
- solution: relacing `scipy.misc.logsumexp` with `scipy.special.logsumexp`

Excutable version pushed to [repo](https://github.com/Meng3www/Recurrent-RSA) instead of submodule in case of potential modifications


------------------------------
### 22.03.2023
issue "beam search and Hyperparameters": original paper uses rationality $\alpha$ = 5.0, beam_width = 10
- beam search generates multiple results, and many of them are not correct English expressions.
- rationality is later used as an index in numpy.ndarray in `bayesian_agents.joint_rsa.py`
    > scores.append(out[world.target,world.rationality,world.speaker])
 
    while non-int cannot be an index. Once it's converted into int, there is an index out of bound error.

issue "Dataset": `utils.test_data.py`
- multiple variables without initiation. possible missing files/folders

More issues (questions):
- missing "charpragcap" directory and many files in author's github repository leading to many files not runnable
- not stated clearly in the paper which data sets in [Visual Genome](http://visualgenome.org/api/v0/api_home.html) are adopted by the author as test sets that generated the testing results
- the current trained models in author's repo are character-level models, might need to train word-level models for the evaluation
- is "lang_mod" a character-level S0 model?
- it seems that the "coco" model is for generating captions and the "vg" model for L_eval (see https://github.com/reubenharry/Recurrent-RSA/issues/4). The "vg" model seems to be a speaker model that behaves similarly to "coco", do we have to apply Bayes' rule to obtain a listener model for evalution (how?)
- as discussed above, "beam search" generates many outcomes that are not correct expressions, but these incorrect expressions only appear in pragmatic caption, not in literal caption. However, according to the author's evaluation, pragmatic caption (S1) should be better than literal caption (S0)??? And greedy seems to perform better than beam search, which is also the opposite to the author's statement
- `recursion_schemes/recursion_schemes.py`  
    \# originally 
    > \>\>\> state.context_sentence   
    [[[1],  [2],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0],  [0]]]    
    
    \# after the following line
    > \>\>\> state.context_sentence=context_sentence    
    \>\>\> state.context_sentence    
    ['^']    
    
    what is the point of getting the original state.context_sentence?


------------------------------
### 25.03.2023
a python tool for retrieving data from Visual Genome found: [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver)
- invalid methods: 
    - api.get_all_image_ids()
    - api.get_image_ids_in_range(start_index=2000, end_index=2010)

Visual Genome API Documentation: http://visualgenome.org/api/v0/api_endpoint_reference.html
- invalid endpoints: 
    - /api/v0/images/all
    - /api/v0/image/{:id}/qa
    - /api/v0/qa/all
    - /api/v0/qa/:q_type


------------------------------
### 27.03.2023
a resnet model that is used in build_data.py is not provided by the author, could not proceed building data


    
    
