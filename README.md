# annaRBM

annaRBM trains Restricted Boltzmann Machines (RBMs) on aligned biological sequences together with their annotations
(e.g. a function, a phenotype or a measured property). The annotations are modeled by a label layer coupled to the hidden
layer, so that a single model

- learns the statistics of the sequences and their relation with the annotations,
- generates new sequences, either freely or conditioned on a target annotation,
- predicts the annotation of a sequence, p(label | sequence),
- uses also the sequences without annotation during training.

<p align="center">
    <img src="images/labelRBM.png" alt="RBM with an annotation layer" width="300"/>
</p>

## Installation

The project is managed with [uv](https://docs.astral.sh/uv/). To install it from source:
```bash
git clone https://github.com/rossetl/annaDCA.git
cd annaDCA
uv sync
```
`uv sync` creates the virtual environment `.venv` with the locked dependencies (`uv.lock`) and the development tools.
Commands are run in this environment with `uv run`, e.g. `uv run annarbm train -h`. The plotting helpers
(`annarbm.plot`) need the `plot` extra: `uv sync --extra plot`.

To use the package in another uv project, add it with `uv add <path or git url>`, or, once released on PyPI,
`uv add annarbm`. A plain `pip install .` from the repository also works.

The only dependencies are PyTorch, NumPy, pandas and tqdm. Models run on CPU, CUDA GPUs and Apple GPUs (MPS).

## Quick start

Train a model on the chorismate mutase sequences of `example_data`, annotated as functional or not:
```bash
uv run annarbm train -d example_data/Natural_CM.csv --column_names header \
    -o models -l cm -H 50 --nchains 1000 --nupdates 2000 --device auto
```
Sample 1000 sequences conditioned on the label `True`:
```bash
uv run annarbm sample -p models/cm_checkpoints/model_update_2000.pt -o cm_true.fasta -n 1000 --target True --device auto
```
The tutorial [`tutorials/use_rbm.ipynb`](tutorials/use_rbm.ipynb) shows how to analyse the trained model.

## Input data

The training needs the sequences, their identifiers and their annotations. They can be provided as

- a single `csv` file (`-d`) with one column for the identifiers, one for the aligned sequences and one for the annotations.
  The default column names are `name`, `sequence` and `label`, and can be changed with `--column_names`,
  `--column_sequences` and `--column_labels`;
- a `fasta` file (categorical variables) or a plain text file with extension `.txt` or `.dat` (binary variables) given
  to `-d`, plus a `csv` file given to `-a` with the identifiers and the annotations. The identifiers must match the fasta
  headers; for text files, the identifier of a sequence is its row index (starting from 0).

**Sequences.**
- Categorical variables: aligned sequences over an alphabet, chosen with `--alphabet` among `protein` (default),
  `rna`, `dna`, or given explicitly as a string of tokens. Sequences with symbols outside the alphabet are discarded.
- Binary variables (`--is_binary`): strings of `0` and `1` in a csv file, or rows of 0/1 values separated by white
  spaces in a text file.

The sequences are reweighted to correct for the sampling bias: the weight of a sequence is the inverse of the number of
sequences with sequence identity larger than `--clustering_seqid` (default 0.8). Use `-w` to provide precomputed weights,
or `--no_reweighting` to give the same weight to all the sequences.

**Annotations.**
- By default, the annotations are categorical: each distinct value is a class.
- With `--continuous_labels`, they are real numbers, standardized and modeled with a Gaussian label layer.
- Missing annotations (empty fields, or sequences absent from the annotation file) are allowed. By default
  (`--missing_labels posterior`), the label of an unannotated sequence is summed over with its posterior
  p(label | sequence), which gives the exact likelihood gradient of these sequences. With `--missing_labels zero`, the
  missing label is replaced by an all-zero vector, which removes the label-hidden coupling for that sequence. Missing
  continuous annotations are set to the mean value.
- Duplicated sequences, or duplicated identifiers in the annotation file, are merged. If they carry different
  annotations, the annotation is set to missing (categorical) or to their mean (continuous).

## Training

```bash
annarbm train -d <data> [-a <annotations>] -o <output folder> -l <model tag> [options]
```
The main options are listed below; `annarbm train -h` lists all of them.

| Option | Default | Description |
| --- | --- | --- |
| `-H`, `--hidden` | 100 | Number of hidden units. |
| `--potential` | `bernoulli` | Type of the hidden units: `bernoulli`, `gaussian` or `relu`. |
| `--nupdates` | 10000 | Number of gradient updates. |
| `--nchains` | 5000 | Number of persistent Markov chains. |
| `--batch_size` | `nchains` | Number of data sequences per update. All the sequences are used, in a new random order at each epoch. |
| `--gibbs_steps` | 10 | Gibbs sampling steps per update. |
| `--lr`, `--lr_final` | 0.01, none | Learning rate. If `--lr_final` is given, it decays to this value with a cosine schedule. |
| `--max_grad_norm` | none | Clipping of the gradient norm. |
| `--pseudocount` | 1/Meff | Regularization of the statistics of the discrete variables (mixture with the uniform distribution). |
| `--l1`, `--l2`, `--l1l2` | 0 | Regularization of the weight matrix (the L1² penalty favors sparse hidden units). |
| `--l1_label`, `--l2_label` | 0 | Regularization of the label matrix. |
| `--init_from_data` | off | Initialize the visible and label biases with the independent-site model of the data. |
| `--no_standardize`, `--no_centering` | off | Gradient type, see below. |
| `--device` | `cuda` | `cuda`, `mps`, `cpu` or `auto`. Falls back to CPU if CUDA is not available. |

**Gradient.** By default, the gradient is *standardized*: it is computed in a parametrization of the model where the
visible, hidden and label units are centered on their data mean, and the hidden and label units are rescaled by their
standard deviation. This makes the training faster and more stable. The parameters are always stored in the original
parametrization. `--no_standardize` only centers the units, and `--no_centering` uses the plain gradient.

**Outputs.** In the output folder, the training saves
- `<tag>.log`: the training options and, every `--save_every` updates, the learning rate, the Pearson correlation
  between the connected two-point correlations of the data and of the chains, and the accuracy of the label prediction
  on the annotated sequences;
- `<tag>_checkpoints/model_update_<n>.pt`: self-describing checkpoints with the model configuration and parameters, the
  chains (in compact form), the optimizer state, the alphabet and the label names;
- `<tag>_weights.dat`: the sequence weights.

The training stops if the outputs of the same tag already exist. Use `--overwrite` to replace them, or
`-p <checkpoint>` to resume the training from a checkpoint (the log file is then extended).

## Sampling

```bash
annarbm sample -p <checkpoint> -o <output.fasta> -n 1000 --gibbs_steps 1000 [--target <label>] [--beta 1.0]
```
The sequences are saved in fasta format with the label in the header (`>chain_<i>|label=<label>`). With `--target`, the
sequences are sampled conditioned on the label (a class name, or a value for continuous annotations); otherwise, the
labels are sampled together with the sequences.

## Python API

```python
from annarbm import AnnaDataset, get_rbm, load_rbm

dataset = AnnaDataset("example_data/Natural_CM.csv", column_names="header")
rbm = get_rbm(
    visible_type="potts",          # 'potts', 'bernoulli', 'gaussian', 'relu'
    hidden_type="bernoulli",       # 'bernoulli', 'gaussian', 'relu'
    visible_shape=(dataset.L, dataset.q),
    hidden_shape=100,
    num_classes=dataset.get_num_classes(),
)
samples = rbm.sample(gibbs_steps=100, num_samples=1000)                    # visible, hidden and label chains
conditioned = rbm.sample_conditioned(gibbs_steps=100, targets=dataset.labels)
p_labels = rbm.predict_labels(dataset.data_one_hot)                        # p(label | sequence)

# A trained model, with its chains and metadata
rbm, checkpoint = load_rbm("models/cm_checkpoints/model_update_2000.pt")
chains, metadata = checkpoint["chains"], checkpoint["metadata"]
```

## Model

The energy of a configuration of visible units v (the sequence), hidden units h and label l is

E(v, h, l) = U_v(v) + U_h(h) + U_l(l) - vᵀ W h - lᵀ L h,

where W is the weight matrix, L the label matrix, and U_v, U_h, U_l the potentials of the layers:

| Layer | Units | Use |
| --- | --- | --- |
| `PottsLayer` | categorical, one-hot | visible (sequences) |
| `BernoulliLayer` | {0, 1} | visible or hidden |
| `GaussianLayer` | real | visible, hidden or continuous label |
| `ReLULayer` | real, non-negative | visible or hidden |
| `CategoricalLabelLayer` | categorical, one-hot | categorical label |

Each layer is a subclass of `annarbm.layers.Layer`. New types of units are added by implementing its abstract
methods; a subclass that defines `layer_type` is registered, so that it can be saved in and loaded from checkpoints.

## Development

```bash
uv run pytest
```
The tests check the gradients, energies and samplers against exact computations on small models, the data parsing, and
the command line workflows.

## License

Apache 2.0, see [LICENSE](LICENSE).
