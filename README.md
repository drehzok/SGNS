## SGNS
Forked from zh3nis/SGNS

### Requirements
Code is written in Python 3.6 and requires Pytorch 1.3+. It also requires the following Python modules: `numpy`, `gensim`, `argparse`, `gdown`. You can install them via:
```bash
pip install numpy gensim argparse gdown
```

### Model
To reproduce the SGNS results from Table 1
```
mkdir embeddings
python main.py --valid data/valid.txt
python eval.py
```
