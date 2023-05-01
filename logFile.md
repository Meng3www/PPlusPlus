### 26.04.2023
test run on urobe (cpu)
|epochs|loss               |time     |
| ---- | ----------------- | ------- |
|1     |739414.5242900848  |885.8389172554016(14.76mins)|

model saved and loaded to continue training
|epochs|loss               |time     |
| ---- | ----------------- | ------- |
|1     |620580.0077550262  |873.3544211387634|
|1     |619390.9568524957  |888.3745331764221|

screen `vg_word`, 200 epoch, eta -, lr=0.0005      
>(pre-trained) model loaded, resume training every 40 epoch

|epochs|loss    |time/m|epochs|loss    |time/m|loss    |time/m|epochs|loss    |time/m|
| ---- | ------ | ---- | ---- | ------ | ---- | ------ | ---- | ---- | ------ | ---- |
|1     |738076  |14    |41    |171230  |22    |125789  |22    |81    |16372   |23    |
|2     |609680  |30    |42    |167607  |46    |101464  |44    |82    |13955   |47    |
|3     |555475  |45    |43    |167964  |69    |92259   |68    |83    |13080   |71    |
|4     |512368  |62    |44    |167989  |93    |86531   |90    |84    |12391   |95    |
|5     |469827  |82    |45    |166317  |120   |82123   |113   |85    |11767   |118   |
|6     |434377  |104   |46    |166167  |148   |78152   |135   |86    |11217   |143   |
|7     |402436  |126   |47    |164424  |176   |75686   |158   |87    |10745   |167   |
|8     |372768  |149   |48    |164851  |203   |73069   |180   |88    |10307   |191   |
|9     |347840  |172   |49    |163030  |230   |71266   |203   |89    |9858    |215   |
|10    |324123  |195   |50    |162742  |255   |69159   |225   |90    |9501    |239   |
|11    |305901  |216   |51    |162231  |279   |67636   |248   |
|12    |289051  |238   |52    |162297  |306   |65677   |270   |
|13    |275662  |259   |53    |160993  |333   |64813   |293   |
|14    |262105  |281   |54    |161012  |359   |63786   |316   |
|15    |250981  |302   |55    |161087  |385   |62452   |339   |
|16    |240427  |323   |56    |160636  |409   |61329   |362   |
|17    |231964  |345   |57    |161310  |434   |59915   |385   |
|18    |224051  |366   |58    |162233  |458   |59227   |411   |
|19    |217940  |388   |59    |160728  |482   |58834   |437   |
|20    |212230  |410   |60    |163124  |507   |57491   |461   |
|21    |208360  |431   |61    |160421  |531   |45832   |22    |
|22    |202899  |453   |62    |160598  |556   |39115   |46    |
|23    |198813  |474   |63    |161740  |581   |36499   |69    |
|24    |194673  |496   |64    |161535  |607   |34487   |92    |
|25    |191586  |517   |65    |160730  |635   |32918   |115   |
|26    |188432  |539   |66    |161747  |665   |31881   |139   |
|27    |186135  |560   |67    |159504  |695   |30807   |162   |
|28    |183735  |581   |68    |160751  |721   |29555   |186   |
|29    |182308  |602   |69    |159240  |747   |28818   |209   |
|30    |179695  |623   |70    |160499  |773   |28072   |233   |
|31    |177016  |646   |71    |160458  |799   |27188   |257   |
|32    |176084  |668   |72    |160139  |826   |26536   |285   |
|33    |175856  |690   |73    |158902  |852   |25742   |312   |
|34    |173902  |712   |74    |158958  |878   |25413   |338   |
|35    |172035  |735   |75    |162316  |904   |24559   |362   |
|36    |171402  |758   |76    |160553  |929   |24003   |386   |
|37    |170533  |780   |77    |160449  |955   |23707   |410   |
|38    |169336  |802   |78    |160896  |982   |23211   |434   |
|39    |169255  |825   |79    |160457  |1008  |22662   |458   |
|40    |167633  |847   |80    |159899  |1034  |21972   |482   |

col1-3: vg_word_decoder_40.pkl      
col4-6: vg_word_decoder_80_5.pkl     
col7-8: vg_word_decoder_50_3.pkl, vg_word_decoder_60_3.pkl, vg_word_decoder_70_2.pkl, vg_word_decoder_80_2.pkl     
col9-11: vg_word_decoder_90_1.pkl



------------------------------
### 23.04.2023
### training each round with 25 epochs
`lr=0.0005`, `dataloader`: `shuffle=True`

|epochs|loss               |time     |
| ---- | ----------------- | ------- |
|1 	   |739216.6607464552  |174.2415554523468|
|2 	   |610703.1258666664  |177.63561868667603|
|3 	   |556093.274983108 	 |176.83281898498535|
|4 	   |512309.349638246 	 |175.083979845047|
|5 	   |473633.9676604681  |174.0270426273346|
|6 	   |435549.20524051413 |174.2834174633026|
|7 	   |403200.7785284724  |175.86287331581116|
|8 	   |373050.44353656843 |178.4084255695343|
|9 	   |347408.65652307495 |177.00395822525024|
|10    |322396.2263917038  |176.33732533454895|
|11 	 |304676.6200012285  |178.08526396751404|
|12 	 |288814.681029662 	 |175.510089635849|
|13 	 |274578.8468334521  |176.97605872154236|
|14 	 |262650.92392182164 |175.4945158958435|
|15 	 |251759.12047119252 |173.46828198432922|
|16 	 |241935.81245015806 |175.62295389175415|
|17 	 |233111.94927201892 |173.8282699584961|
|18 	 |225285.48874083115 |174.48191022872925|
|19 	 |218314.5251097302  |173.74017810821533|
|20 	 |212438.11016655806 |173.7122197151184|
|21 	 |206609.44695721718 |178.14122653007507|
|22 	 |203140.75414126227 |174.42536234855652|
|23 	 |198690.6035451973  |174.53765892982483|
|24 	 |195167.58255624826 |176.0397870540619|
|25 	 |191166.9267608953  |173.7200322151184|

*saving and then loading the model to continue training cause the loss to increase

### test on the 1st epoch loss with different lr
|lr      |loss  |
| ------ | ---- |
|0.0001  |807939|
|0.0002  |765025|
|0.0003  |748410|
|0.0004  |742096|
|0.0005  |739070|
|0.0006  |738066|
|0.0007  |738964|

### test on the 1st epoch loss with lr = 0.0006
|epochs|loss               |time     |
| ---- | ----------------- | ------- |
|1    |738013.8100529313   |182.07085394859314|
|2    |615120.0524843037   |178.58732175827026|
|3    |566254.86066176 	   |174.16399240493774|
|4    |523824.8683729768   |173.6789710521698|
|5    |488836.4067925513   |173.39287161827087|

### test 10 epoch, with loss and time for each epoch
`lr=0.0005`, `dataloader` shuffles
|epochs|loss            |time|
| ---- | -------------- | ------- |
|1 	 |739070.7953240573 	 |173.92298364639282|
|2 	 |609138.9648354053 	 |176.02179741859436|
|3 	 |555254.1776845008 	 |178.81826758384705|
|4 	 |510954.70532573014 	 |175.1984031200409|
|5 	 |468700.4642474279 	 |175.1154980659485|
|sum     |2883119.10741712144    |879.07694983482363|
|*       |3141720.706070017      |859.8828392028809|
|6 	 |433051.2343964726 	 |179.41459846496582|
|7 	 |398923.7756019421 	 |177.6712441444397|
|8 	 |370741.56857311726 	 |174.9203226566314|
|9 	 |342807.43934043683 	 |176.92594742774963|
|10  	 |321479.41265753005 	 |182.3168785572052|
|sum 	 |1867003.43056949884  	 |891.24899125099175|
|*   	 |2046428.1950347265   	 |855.635754108429|

30 mins/10 epochs
### test 2 epoch, save model, load model, proceed another 2 epoch
|epochs|loss            |time|
| ---- | -------------- | ------- |
|1   	 |739679.3844605982 	|178.96493577957153|
|2 	 |609023.010008961 	|176.55454230308533|

save and load model.state_dict()   

|epochs|loss            |time|
| ---- | -------------- | ------- |
|3 	 |571352.3976150751 	 |180.7889723777771|
|4 	 |516888.27769570425 	 |175.61053657531738|

------------------------------
### 22.04.2023
- test train() lr=0.001:
> n_epoch = 50      
> print_every = 5        

result: 2h21m54s (robot verification pops up around 100m later)      
|epochs	|loss			|time|     
| ------| --------------------- | --------------- |
|5	|3526568.322480902 	|857.7873964309692|     
|10 	|2794232.8348501776 	|850.9735901355743|     
|15 	|2349121.2564896713 	|851.2090680599213|     
|20 	|2126159.436086815 	|848.9444327354431|     
|25 	|2007716.2772798953 	|850.5208220481873|     
|30 	|1942019.1939218743 	|849.3143208026886|     
|35 	|1897457.179702924 	|851.3509232997894|     
|40 	|1874015.2446018406 	|852.029111623764|     
|45 	|1850469.8763161292 	|850.1490111351013|     
|50 	|1833252.8137487597 	|851.7524993419647|     

test train() lr=0.005:
result: manually stopped after 43m1s
|epochs	|loss			|time|     
| ------| --------------------- | --------------- |     
|5 	|4937965.998798579 	|855.2068405151367|     
|10 	|5115216.958911061 	|858.3145492076874|     
|15 	|5394895.657621026 	|857.7459843158722|   

test train() lr=0.0005:
result: 1h8m41s
|epochs	|loss			|time|     
| ------| --------------------- | --------------- |     
|5 	|3141720.706070017 	|859.8828392028809|     
|10 	|2046428.1950347265 	|855.635754108429|     
|15 	|1557936.1527801761 	|856.0033431053162|     
|20 	|1345679.0653578276 	|856.8627445697784|   

forced to stop for `cannot currently connect to a GPU due to usage limits in Colab`

- coco vocab pickled: coco_data/vocab_coco_4533.pkl, coco_data/vocab_coco_6336.pkl


------------------------------
### 19.04.2023
- the encoders and decoders provided by the author are different, even though some have the same size
- vg-encoder-5-3000.pkl borrowed to get features from image
- getTrainingPair() as generator is very slow (10 pairs/5 sec), if used directly during the training, it could last a long time and time out more than needed. 
- [Datasets & DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#datasets-dataloaders):     
pre-process dataset and get all features and captions     
append batches of the results in a pkl     
create own dataset class and use dataloader


------------------------------
### 15.04.2023
- vocab pickled: vg_data/vocab.pkl (14286), vg_data/vocab_small.pkl (4987)
- try out exisiting models with seg_type = 'word' 
current models provided by the authors do not work with seg_type = 'word' for below error:
> RuntimeError: Error(s) in loading state_dict for DecoderRNN:      
	size mismatch for embed.weight: copying a param with shape torch.Size([30, 256]) from checkpoint, the shape in current model is torch.Size([14286, 256]).     
	size mismatch for linear.weight: copying a param with shape torch.Size([30, 512]) from checkpoint, the shape in current model is torch.Size([14286, 512]).     
	size mismatch for linear.bias: copying a param with shape torch.Size([30]) from checkpoint, the shape in current model is torch.Size([14286]).

see more in `seg_type_word` branch. branch will not be updated further. more update: see `word_model_branch`


------------------------------
### 14.04.2023
- try out exisiting models with seg_type = 'word'
	
- eval: put image to coco and get chars0, chars1 greedy     
put the same image to vg and get chars0, chars1 beam?     
compare prob. what if coco s0 and s1 not in vg s0 s1?     

- use ts1 only, read local image 


------------------------------
### 27.03.2023
a resnet model that is used in build_data.py is not provided by the author, could not proceed building data


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
### 01.02.2023
**priority** code, data set up and running

**to-dos**
- make the code our own (with modification and documentation)
- run some examples before training, eg. for word base beam search
- t-test is sufficient 
- performance metrics


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


----------------------------------
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
