Andreas & Klein 2016 Reasoning about Pragmatics
----------------------
In their paper "Reasoning about Pragmatics with Neural Listeners and Speakers", Jacob Andreas and Dan Klein of the University of California, Berkeley present a model for pragmatically describing scenes, in which contrastive behavior results from a combination of inference-driven pragmatics and learned semantics
. The paper examines the pragmatic meaning of comparative constructions from a neural perspective
, and evaluates language generation and interpretation to show that pragmatic inference improves state-of-the-art listener models and speaker models in correctly generating and following natural language
.The paper builds on speaker and listener models that reason iteratively and counterfactually about instruction sequences, focusing on instruction following and instruction generation tasks
. It also explores whether there is interpretable logical and compositional structure in language learning
. The general theme of the paper is pragmatic reasoning and language understanding through interaction
.  
The neural network used is a listener net which is used to obtain the correct image over a set of candidate images
. It is also used to systematically manipulate human language acquisition by adopting neural networks (NN).

Vedantam et al 2017 Context-aware Captions
---------------------
The paper "Context-Aware Captions from Context-Agnostic Supervision" introduces an inference technique to produce discriminative context-aware image captions (captions that describe differences between images or visual concepts) using only generic context-agnostic supervision .  
The authors focus on deriving pragmatic (context-aware) behavior given access only to generic context-agnostic data
.The paper describes an approach for inducing context-aware language for justification, where the context is another class, and discriminative image captioning, where the context is a semantically related concept
. The authors first train a generic context-agnostic image captioning model (the "speaker") using training data from Reed et al. who collected captions describing bird images on the web.  
The paper acknowledges some fundamental limitations to inducing context-aware captions from context-agnostic supervision, such as if two distinct classes are visually similar but have different meanings
. The key contributions of this paper are: 1) introducing an inference technique to produce discriminative context-aware image captions using only generic context-agnostic supervision; 2) describing an approach for inducing context-aware language for justification and discriminative image captioning; and 3) acknowledging some fundamental limitations to inducing context-aware captions from context-agnostic supervision.


Cohn-Gordon et al 2018 Incremental Iterated Response Model
---------------------
Christopher Potts' "An Incremental Iterated Response Model of Pragmatics" is a game theoretic model of language use and interpretation that explains pragmatic phenomena as arising from agents reasoning about each other to increase communicative efficiency  
It  conceptualizes pragmatic reasoning as a recursive process in which the speaker's utterance is iteratively modified based on the listener's response
. The model proposes that this process is incremental, meaning that the speaker's utterance is modified one step at a time, rather than all at once
. This paper provides an analysis of how this model can be used to explain various aspects of human communication, such as how speakers adjust their utterances to fit the context and how listeners interpret ambiguous messages

Nie et al 2020 Pragmatic Issue
---------------------
The paper focuses on the task of image captioning and how to make it more sensitive to pragmatic issues. It proposes a new approach to image captioning that takes into account the context of an image, such as its location or time period, in order to generate more accurate captions. The authors also discuss how their approach can be used in other tasks such as scene description-to-depiction tasks
.The paper presents a new dataset called DynaSent which consists of images with associated captions that are annotated with pragmatic information. The authors then use this dataset to evaluate their proposed approach and show that it outperforms existing methods in terms of accuracy and robustness. They also demonstrate how their approach can be used to generate more accurate captions for images with complex contexts.