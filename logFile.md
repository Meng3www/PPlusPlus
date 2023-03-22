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
error "Module 'scipy.misc' has no attribute 'logsumexp'" bayesian_agents.joint_rsa.py
- solution: relacing `scipy.misc.logsumexp` with `scipy.special.logsumexp`
Excutable repo pushed to https://github.com/Meng3www/Recurrent-RSA instead of submodule in case of potential modifications
