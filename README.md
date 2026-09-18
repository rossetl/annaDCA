# annaDCA

Annotation Assisted Direct Coupling Analysis (annaDCA) trains restricted Boltzmann
machines (RBMs) on aligned sequences and annotations, predicts labels, and generates
sequences conditioned on a label. It supports categorical alphabets (proteins,
DNA, RNA, or custom tokens) and binary data using PyTorch.

## Installation

Requires Python 3.11 or newer. Install a published release in your uv project:

```bash
uv add annadca
uv run annadca --help
```

For local development:

```bash
git clone https://github.com/rossetl/annaDCA.git
cd annaDCA
uv sync
uv run annadca --help
```

Alternatively, use `uv pip install annadca` or `python -m pip install annadca`
inside a virtual environment. Optional extras are `annadca[ptt]` for importing
`rbms` checkpoints and `annadca[plot]` for plotting helpers. For example,
`uv sync --extra ptt` enables checkpoint import in a development checkout.

The CLI defaults to CUDA; pass `--device cpu` for CPU execution. PyTorch's default
installation can include large accelerator dependencies.

## Input data

The simplest input is a CSV with unique identifiers, aligned sequences of equal
length, and one label per row:

```csv
name,sequence,label
seq1,ACDE,family_a
seq2,ACDF,family_b
seq3,ACDG,family_a
```

Use `--column_names`, `--column_sequences`, and `--column_labels` to select other
column names (case sensitive). Identifiers, sequences, and labels are read as
strings, preserving leading zeros and labels such as `NA`. Empty labels represent
missing annotations; training requires at least one annotated row, and sampling
requires every row to have a known label.

Alternatively, supply an aligned FASTA with `-d alignment.fasta` and an annotation
CSV containing `name,label` with `-a annotations.csv`. Identifiers must match the
FASTA headers. For binary data, pass `--is_binary` and either a CSV with strings
such as `0101`, or a whitespace-separated `.txt`/`.dat` matrix with annotations
whose identifiers are zero-based row numbers (`0`, `1`, ...).

The default alphabet is `protein`. Use `--alphabet dna`, `--alphabet rna`, or a
custom string such as `--alphabet AC` to select another alphabet. Token order
must stay consistent between training and sampling.

## Training

```bash
uv run annadca train -d sequences.csv -o outputs -l example \
  --hidden 100 --nchains 100 --nepochs 1000 --gibbs_steps 10 --device cpu
```

`--nepochs` counts gradient updates. Training uses persistent contrastive
divergence. Sequence weights are computed by default; use `--no_reweighting` for
unit weights or `--weights weights.dat` for one nonnegative weight per row.
Other controls include `--lr`, `--eta`, `--l1`, `--l2`, `--uncentered`,
`--init_from_profile`, `--seed`, and `--dtype`.

Pass `--marginalize_missing_labels` to sum over missing annotations during training
(binary and categorical models). Annotated rows use `log p(sequence, label)`;
unannotated rows use `log sum_label p(sequence, label)`. The positive phase uses
exact posterior label, hidden, and joint label–hidden expectations; the persistent
negative phase is unchanged. This also works with centered updates and sequence
weights; the existing pseudocount, regularization, and `--eta` settings still apply.
Without the flag, missing annotations retain the existing zero-vector treatment.
Marginalization needs additional work and memory proportional to the number of
missing rows times the number of labels times the number of hidden units.
For Python training, use `pcd(..., marginalize_labels=True)`. Specify the flag
again when resuming training; the selected mode is recorded in the training log.

Outputs are `example_params.h5`, `example_chains.fasta`, and `example.log`, plus
computed weights when enabled. Without `-l`, the names are `params.h5`,
`chains.fasta`, and `annaRBM.log`. The parameter file stores the latest checkpoint;
each save replaces the previous checkpoint. Completed CLI training also records
the label vocabulary and alphabet for sampling.

Resume with `-p outputs/example_params.h5 -c outputs/example_chains.fasta`, using
the same dataset, alphabet, and label vocabulary, and a larger `--nepochs` target.
Use a new output directory to retain the previous run. With the `ptt` extra,
`--path_params_ptt model.h5` initializes annotated training from an `rbms`
checkpoint instead of `-p`/`-c`.

## Conditioned sampling

```bash
uv run annadca sample -d targets.csv -p outputs/example_params.h5 \
  -o samples.csv --gibbs_steps 1000 --device cpu
```

The target CSV uses the same columns and aligned sequence format as training.
Each row requests one new sequence conditioned on its label; its input sequence
is used for dataset validation, not as the starting chain. Output is a CSV with
new numeric identifiers, generated sequences, and conditioning labels. Labels
may be a subset of the training vocabulary for checkpoints produced by the
updated training CLI. Older checkpoints do not store the vocabulary: provide all
original training labels, using exactly their original names, so their sorted
encoding matches training. Unknown and missing labels are rejected.

Run `annadca train --help` or `annadca sample --help` for all options.
`python -m annadca` provides the same interface.

## Python API

```python
from annadca import annaRBMcat
from annadca.dataset import annaDataset

# Use the training data to recover its full label encoding.
data = annaDataset("sequences.csv", alphabet="protein")
model = annaRBMcat()
model.load("outputs/example_params.h5")
probabilities = model.predict_labels(data.data_one_hot)
predictions = data.to_label(probabilities)
targets = data.to_one_hot(["family_a", "family_b"])
chains = model.sample_conditioned(gibbs_steps=100, targets=targets)
```

`annaRBMbin` exposes the corresponding binary model. Models provide parameter
initialization, Gibbs sampling, label prediction, energy calculations, and
checkpoint I/O. `annaDataset` exposes encoded sequences, labels, weights, and
label conversion helpers. `annadca.train.pcd` performs a gradient-estimation step
for custom training loops. Categorical visible tensors have shape `(N, L, q)`;
binary tensors have shape `(N, L)` and label tensors `(N, K)`.

## Development and release

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv build
uv run twine check dist/annadca-0.2.2*
```

Builds produce a wheel and source archive in `dist/`. Version metadata lives in
`pyproject.toml`; use a new version for every PyPI release. After reviewing and
testing the artifacts, publish only the intended version with `uv publish` and
explicit artifact paths. Publishing is a separate maintainer step.

`project-specificity` currently pins `annadca==0.2.2` with a local uv source.
Once that version is published, remove its `annadca` path entry from
`[tool.uv.sources]` and run `uv lock` followed by `uv sync`. Its existing `rbms`
dependency supplies PTT support; other consumers should install `annadca[ptt]`.

Licensed under Apache-2.0; see [LICENSE](LICENSE).

### Optimized sampling and optional Triton backend

`sample` and `sample_conditioned` use a shared PyTorch Gibbs loop that caches the
fixed label projection for conditioned sampling and avoids unused initial tensors.
Supplying both visible and label states also avoids random chain initialization.
Gibbs sampling requires `gibbs_steps >= 1`.

For float32 CUDA models, install the optional backend with
`pip install '.[triton]'` and select it explicitly:

```python
chains = model.sample(10, visible=visible, label=label, backend="triton")
conditioned = model.sample_conditioned(10, targets=label, visible=visible, backend="triton")
hidden = model.sample_hiddens(visible, label, backend="triton")
```

`sample_visibles` and `sample_labels` also accept `backend="triton"`. The default
is `backend="torch"`, supporting existing devices and dtypes. Triton fuses bias,
inverse temperature, probability calculation, and sample output; dense matrix
products still use PyTorch. Categorical kernels support up to 4096 states per
variable. This backend is for sampling and does not provide autograd derivatives.
It uses PyTorch's CUDA RNG and supports reseeding/state restoration, but does not
produce the same random draws as the PyTorch backend.

See [sampling benchmark results and methodology](benchmarks/README.md). Reproduce
on a CUDA machine with:

```bash
python -m pytest tests/test_sampling.py -q
python -m benchmarks.sampling --device cuda --repeats 60 --output benchmarks/results_cuda.json
```

Training enables Triton by default (`--triton`). Use `annadca train ... --no-triton`
for PyTorch sampling, including CPU or float16/float64 training. Triton training
requires a CUDA device, float32, and the optional Triton dependency. The selected
backend is printed at startup and recorded in the training log. This CLI default
does not change the Python sampling API's default backend.
