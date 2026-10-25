import argparse
import os
from typing import List, Optional

import torch

from annarbm.io import save_chains
from annarbm.layers.label import CategoricalLabelLayer
from annarbm.parser import add_args_sample
from annarbm.rbm import load_rbm
from annarbm.utils.sequences import get_device, get_dtype


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annarbm sample", description="Sample sequences from an annaRBM model.")
    return add_args_sample(parser)


def _target_labels(rbm, metadata: dict, target: str, num_samples: int) -> torch.Tensor:
    """Label tensor of the target, in the representation used by the model."""
    if isinstance(rbm.label_layer, CategoricalLabelLayer):
        names = {str(name): idx for idx, name in metadata.get("idx_to_label", {}).items()}
        if target in names:
            idx = names[target]
        elif target.isdigit() and int(target) < rbm.num_classes and not names:
            idx = int(target)
        else:
            raise ValueError(f"Unknown target label '{target}'. Available labels: {list(names) or list(range(rbm.num_classes))}")
        labels = torch.zeros(num_samples, rbm.num_classes)
        labels[:, idx] = 1.0
    else:
        value = (float(target) - metadata.get("labels_mean", 0.0)) / metadata.get("labels_std", 1.0)
        labels = torch.full((num_samples, 1), value)
    return labels.to(rbm.weight_matrix)


def main(argv: Optional[List[str]] = None):
    args = create_parser().parse_args(argv)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    torch.manual_seed(args.seed)

    rbm, checkpoint = load_rbm(args.checkpoint, device=device, dtype=dtype)
    metadata = checkpoint.get("metadata", {})
    tokens = metadata.get("tokens")
    if tokens is None:
        raise ValueError("The checkpoint does not contain the alphabet of the sequences.")
    print(f"Loaded model {rbm} from {args.checkpoint} (update {checkpoint['update']}).")

    with torch.no_grad():
        if args.target is not None:
            labels = _target_labels(rbm, metadata, args.target, args.num_samples)
            samples = rbm.sample_conditioned(gibbs_steps=args.gibbs_steps, targets=labels, beta=args.beta)
            samples["label"] = labels
        else:
            samples = rbm.sample(gibbs_steps=args.gibbs_steps, num_samples=args.num_samples, beta=args.beta)

    if not isinstance(rbm.label_layer, CategoricalLabelLayer):
        # Back to the original scale of the continuous labels
        samples["label"] = samples["label"] * metadata.get("labels_std", 1.0) + metadata.get("labels_mean", 0.0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_chains(
        args.output,
        samples,
        tokens=tokens,
        idx_to_label=metadata.get("idx_to_label") or None,
        continuous_labels=not isinstance(rbm.label_layer, CategoricalLabelLayer),
    )
    print(f"{args.num_samples} sequences saved in {args.output}")


if __name__ == "__main__":
    main()
