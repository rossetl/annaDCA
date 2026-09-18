import argparse
import json
import os
import warnings
from typing import List

import h5py
import numpy as np
import pandas as pd
import torch
from adabmDCA.utils import get_device, get_dtype

from annadca import annaRBMbin, annaRBMcat
from annadca.dataset import annaDataset
from annadca.parser import add_args_sample


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample sequences from an annaRBM model.")
    parser = add_args_sample(parser)
    return parser


def _as_token_list(tokens: str | List[str]) -> List[str]:
    if isinstance(tokens, str):
        return list(tokens)
    return list(tokens)


def _decode_sequences(
    visible: torch.Tensor,
    tokens: str | List[str],
    is_binary: bool,
) -> List[str]:
    token_list = _as_token_list(tokens)
    if is_binary:
        idx = (visible > 0.5).to(torch.int64).cpu().numpy()
    else:
        idx = torch.argmax(visible, dim=-1).to(torch.int64).cpu().numpy()
    return ["".join(token_list[i] for i in row) for row in idx]


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.gibbs_steps <= 0:
        parser.error("--gibbs_steps must be positive")

    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    for path in [args.data, args.path_params]:
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Input file {path} not found.")

    torch.manual_seed(args.seed)

    dataset = annaDataset(
        path_data=args.data,
        path_ann=None,
        column_names=args.column_names,
        column_sequences=args.column_sequences,
        column_labels=args.column_labels,
        is_binary=args.is_binary,
        alphabet=args.alphabet,
        no_reweighting=True,
        device=device,
        dtype=dtype,
    )

    if args.is_binary:
        rbm = annaRBMbin()
    else:
        rbm = annaRBMcat()

    upd = rbm.load(
        filename=args.path_params,
        device=device,
        dtype=dtype,
        set_rng_state=False,
    )
    print(f"Model parameters at update {upd} loaded from {args.path_params}")

    with h5py.File(args.path_params, "r") as archive:
        if "alphabet" in archive.attrs and archive.attrs["alphabet"] != dataset.tokens:
            raise ValueError("Sampling alphabet does not match the training alphabet.")
        if "label_vocabulary" in archive.attrs:
            vocabulary = json.loads(archive.attrs["label_vocabulary"])
            dataset.idx_to_label = dict(enumerate(vocabulary))
            dataset.label_to_idx = {label: i for i, label in enumerate(vocabulary)}
            dataset.labels_one_hot = dataset.to_one_hot(dataset.data[args.column_labels].tolist())
        else:
            warnings.warn(
                "Legacy checkpoint has no label vocabulary; provide all training labels with their original names."
            )

    targets = dataset.labels_one_hot
    if targets.numel() == 0:
        raise ValueError("No labels found in the input data file.")

    missing_labels = (targets.sum(dim=-1) == 0).any().item()
    if missing_labels:
        raise ValueError("Conditioned sampling requires a label for every row.")

    if dataset.get_num_classes() != rbm.num_classes():
        raise ValueError(
            "Sampling CSV must contain the same complete label vocabulary as training."
        )
    if dataset.L != rbm.num_visibles() or (not args.is_binary and dataset.q != rbm.num_states()):
        raise ValueError("Sampling sequences and alphabet must match the model dimensions.")

    cond_chains = rbm.sample_conditioned(
        gibbs_steps=args.gibbs_steps,
        targets=targets,
    )

    sequences = _decode_sequences(
        visible=cond_chains["visible"],
        tokens=dataset.tokens,
        is_binary=args.is_binary,
    )
    labels = dataset.to_label(targets)
    names = np.arange(len(sequences)).astype(str).tolist()

    out_df = pd.DataFrame(
        {
            args.column_names: names,
            args.column_sequences: sequences,
            args.column_labels: labels,
        }
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Conditioned samples saved to {args.output}")


if __name__ == "__main__":
    main()
