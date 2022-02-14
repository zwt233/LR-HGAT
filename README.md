# This is the official implementation of LR-HGAT

## Usage

#### Train and evaluate a model:

Define the training parameters by setting the `configs` dictionary in `config_file.py`, and run `CUDA_VISIBLE_DEVICES=0 python main.py`.

## Scripts

- `config_file.py`: file with all configurable parameters.
- `data_process.py`: script processing both training data and testing data.
- `main.py`: script to run experiments (node classification - DBLP dataset).
- `model.py`: script where the model architecture is defined.

## Folders structure

- `/data`: folder containing all intermediate data files.
- `/tmp_models`: where the results of the experiments are stored.

## References:
```
@article{Liu2018efficient,
  title={Efficient Low-rank Multimodal Fusion with Modality-Specific Factors},
  author={Liu, Zhun and Shen, Ying and Lakshminarasimhan, Varun Bharadhwaj and Liang, Paul Pu and Zadeh, Amir and Morency, Louis-Philippe},
  journal={arXiv:1806.00064},
  year={2018}
}
```
```
@inproceedings{zhang2019heterogeneous,
  title={Heterogeneous graph neural network},
  author={Zhang, Chuxu and Song, Dongjin and Huang, Chao and Swami, Ananthram and Chawla, Nitesh V},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={793--803},
  year={2019}
}
```
