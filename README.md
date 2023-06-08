## Attention !!
This is a bit modified version of Tab Transformer, Checkout the "lucidrains/tab-transformer-pytorch" for the original one. This version utilizes the transformer calculation capacity even in the abscence of the categorical features.

## The Original Tab Transformer 
 
Implementation of <a href="https://arxiv.org/abs/2012.06678">Tab Transformer</a>, attention network for tabular data, in Pytorch. This simple architecture came within a hair's breadth of GBDT's performance.
<p align="center"><img src="./tab.png" width="300px"></img></p>

## Install

- First clone this repository in your local machine using git clone commmand and make sure that the original package (lucidrains) is not currently installed on your local system.

```bash
$ pip uninstall tab-transformer-pytorch
```

- Navigate to the setup.py file directory and run the following command:

```bash
$ python setup.py install
```

## Usage

```python
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer


model = TabTransformer(
    num_continuous=1000,                 # Number of continuous features (excluding the label column)
    dim=32,                              # Dimension, you can modify it as per your needs
    dim_out=1,                           # Number of output dimensions (e.g., for binary prediction)
    depth=6,                             # Depth, adjust it based on your dataset complexity
    heads=8,                             # Number of attention heads
    attn_dropout=0.1,                    # Post-attention dropout
    ff_dropout=0.1,                      # Feed-forward dropout
    mlp_hidden_mults=(4, 2),             # Relative multiples of each hidden dimension of the last MLP to logits
    mlp_act=None,                        # Activation for the final MLP (You can manually add your own activation function directly to the output)
)

# assuming that your x_cont is a pandas dataframe
x_cont = torch.tensor(x_cont.values, dtype=torch."your DataType")

pred = model(x_cont) 
```

## Modifications:
- Removed categories Tuple
- Removed categorical features from forward() from TabTransformer Class.
- Removed categorical tokens
- Removed Categorical Constraints
- Removed  Mean-STD Normalization
- Changed the Activation Function from ReLU to Sigmoid (Our task is binary classification)

## Todo

- [ ] consider https://arxiv.org/abs/2203.05556

## Citations

```bibtex
@misc{huang2020tabtransformer,
    title   = {TabTransformer: Tabular Data Modeling Using Contextual Embeddings},
    author  = {Xin Huang and Ashish Khetan and Milan Cvitkovic and Zohar Karnin},
    year    = {2020},
    eprint  = {2012.06678},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{Gorishniy2021RevisitingDL,
    title   = {Revisiting Deep Learning Models for Tabular Data},
    author  = {Yu. V. Gorishniy and Ivan Rubachev and Valentin Khrulkov and Artem Babenko},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2106.11959}
}
```
