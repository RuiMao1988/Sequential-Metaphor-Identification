# Sequential-Metaphor-Identification
Code for the paper "End-to-End Sequential Metaphor Identification Inspired by Linguistic Theories"

https://www.aclweb.org/anthology/papers/P/P19/P19-1378/

## Embeddings
Download glove.840B.300d.zip (https://nlp.stanford.edu/projects/glove/) into glove folder.

Please email r03rm16@abdn.ac.uk for ELMo embeddings, putting them into elmo folder.

(The used dataset and ELMo embeddings were cleaned and pre-trained by https://github.com/gao-g/metaphor-in-context.)

## Environment
python 3.6

pytorch 0.4.1
```
pip install -r requirements.txt
```
## Run
Go to the model folder (rnn_hg or rnn_mhca), then run

VU Amsterdan dataset
```
python main_vua.py
```
Mohammad dataset
```
python main_mohx.py
```
TroFi dataset
```
python main_trofi.py
```
## Citation
```
@inproceedings{mao2019metaphor,
  title={End-to-End Sequential Metaphor Identification Inspired by Linguistic Theories},
  author={Mao, Rui and Lin, Chenghua and Guerin, Frank},
  booktitle={Proceedings of the 57th Conference of the Association for Computational Linguistics},
  pages={3888--3898},
  year={2019}
}
```
