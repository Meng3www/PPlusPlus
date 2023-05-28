# NPNLG Team Project

## Table of contents
* [General info](#general-info)
* [Team members](#team-members)
* [Documentation](#documentation)
* [Project Outcome](#project-outcome)
* [Other Reference](#other-reference)

## General info
This project aims to reproduce the key results of the paper "Cohn-Gordon et al. (2018), Pragmatically Informative Image Captioning with Character-Level Inference" and critically access its evaluation approaches with beam search and greedy sampling for the character- and word-level incremental predictions.   
It is part of the course work for the 6-ECTS course "Universität Tübingen, WS2022/2023, Michael Franke, Neural Pragmatic Natural Language Generation"

## Team members
- [Fanyi Meng](https://github.com/Meng3www)
- [Jia Sheng](https://github.com/jiasheng1100)

## Documentation
General:
- [report/report.pdf](https://github.com/Meng3www/PPlusPlus/blob/main/report/report.pdf): project report
- [logFile.md](https://github.com/Meng3www/PPlusPlus/blob/main/logFile.md): project timeline and progress

Model Training:
- [](): 

Evaluation:
- [evaluate/build_test_data.py](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/build_test_data.py): build Test set 1 data
- [vg_data/ts1_img/](https://github.com/Meng3www/PPlusPlus/tree/main/vg_data/ts1_img): TS1 data
- [evaluate/generate_captions.py](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/generate_captions.py) & [evaluate/generate_captions_word.py](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/generate_captions_word.py): generate captions of TS1 data with char-level and word-level models
- [evaluate/ts1_captions](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/ts1_captions) & [evaluate/ts1_captions_word](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/ts1_captions_word): char- and word-level model-generated captions of TS1 data
- [evaluate/get_accuracy_ts1.py](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/get_accuracy_ts1.py) & [evaluate/get_accuracy_ts1_word.py](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/get_accuracy_ts1_word.py): run automatic evalution to obtain accuracy scores
- [evaluate/get_accuracy_ts1_output.txt](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/get_accuracy_ts1_output.txt) & [evaluate/get_accuracy_ts1_word_output.txt](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/get_accuracy_ts1_word_output.txt): detailed output of evaluation
- [evaluate/ts1_accuracy](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/ts1_accuracy) & [evaluate/ts1_accuracy_word](https://github.com/Meng3www/PPlusPlus/blob/main/evaluate/ts1_accuracy_word): accuracy scores	


## Project Outcome
Our results:

![table1](report/table1.png)

Results from the original papaer:

![table2](report/table2.png)

For more details, please refer to our [project report](https://github.com/Meng3www/PPlusPlus/blob/main/report/report.pdf).

## Other Reference
- Course homepage: https://michael-franke.github.io/npNLG/
- Gohn-Gordon et al. (2918): https://aclanthology.org/N18-2070/


Lastly modified on May 28, 2023\
Tübingen

