# ANNotation Assisted Direct Coupling Analisis (annaDCA)
This package contains the methods and scripts to train and sample an RBM model provided with data annotations.

<p align="center">
    <img src="images/labelRBM.png" alt="RBM Model with Annotations" width="300"/>
</p>

## ⬇️ Installation
### Option 1: from PyPI
```bash
python -m pip install annadca
```
### Option 2: cloning the repository
```bash
git clone https://github.com/rossetl/annaDCA.git
cd annaDCA
python -m pip install .
```

## 📘 Usage
After installation, all the main routines can be launched through the command-line interface using the command `annadca`. For now, only the `train` routine is implemented
To see all the training options do
```bash
annadca train -h
```

### ✔️ Input data format

The training routine requires three types of information: the __training sequences__, their __identifiers__ and the assocated __annotations__.

The package supports both binary variables and categorical variables (e.g. amino acid sequences).

These input data can be provided to the routine in two different ways:

- By providing a `csv` file to the `-d` argument that contains identifiers, sequences and annotations. By default, the routine will check for the columns `name`, `sequence` and `label`, but the user can specify different names using the arguments `--column_names`, `--column_sequences` and `--column_labels`.

- By providing a `fasta` file (categorical variables) or a plain text file with extension `.txt` or `.dat` (binary variables) to the `-d` argument and an additional `csv` file to the `-a` argument containing the sequence identifiers and the annotations. Importantly, the sequence identifiers must match with those that are present in the `fasta` file. For plain text files, the identifier of each sequence is its row index (starting from 0).

To train the model with default arguments and a single csv file, use
```bash
annadca train -d <path_data> -o <output_directory> -l <model_tag>
```

#### Sequence data format
The model supports the following input data format:
- __Binary variables__ (use the `--is_binary` flag):
    - plain text format. Each row is one data sample, variables are separated by white spaces.
    - `csv` format: each sequence is a string of `0` and `1`.
- __Categorical variables__: 
    - [fasta](https://en.wikipedia.org/wiki/FASTA_format) format. Each data poin is a sequence of tokens with an header on top. The header row starts with `>`.
    - `csv` format: the file must contain one column with the aligned training sequences.

#### Annotations
By default, the annotations are treated as categorical: each distinct value of the label column is a class. Missing annotations are allowed and are encoded as an all-zero label vector. Use `--continuous_labels` to treat the annotations as real numbers: they are standardized and modeled with a Gaussian label layer.

### ⚙️ Model and training options
- `--potential`: type of hidden units, among `bernoulli`, `gaussian` and `relu`.
- By default, the gradient is __standardized__: the visible, hidden and label units are centered on their data mean, and the hidden and label units are rescaled by their standard deviation, in the parametrization used to compute the gradient. The parameters of the model are always stored in the original parametrization. Use `--no_standardize` to only center the units, or `--no_centering` to use the plain gradient.
- `--l1`, `--l2`, `--l1l2`: regularization of the weight matrix.
- `--no_compile`: disable `torch.compile` for the sampling functions (useful for debugging, or when no compiler toolchain is available).

## 🐍 Python API
```python
from annadca import annaDataset, get_rbm

dataset = annaDataset("example_data/Natural_CM.csv", column_names="header")
rbm = get_rbm(
    visible_type="potts",          # 'potts', 'bernoulli', 'gaussian', 'relu'
    hidden_type="bernoulli",       # 'bernoulli', 'gaussian', 'relu'
    visible_shape=(dataset.L, dataset.q),
    hidden_shape=100,
    num_classes=dataset.get_num_classes(),
)
samples = rbm.sample(gibbs_steps=100, num_samples=1000)                    # visible, hidden and label chains
conditioned = rbm.sample_conditioned(gibbs_steps=100, targets=dataset.labels_one_hot)
p_labels = rbm.predict_labels(dataset.data_one_hot)                        # p(label | visible)
```
Each layer of the model is a subclass of `annadca.layers.Layer`, so that new types of units can be added by implementing the abstract methods of this class.

## 🧪 Tests
```bash
python -m pytest tests
```
