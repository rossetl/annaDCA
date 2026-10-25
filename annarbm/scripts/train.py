import argparse
import math
import os
import shutil
import time
import warnings
from typing import List, Optional

import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm

from annarbm.dataset import AnnaDataset
from annarbm.parser import add_args_train
from annarbm.rbm import get_rbm, load_rbm, save_checkpoint
from annarbm.utils.analysis import label_accuracy, two_point_pearson
from annarbm.utils.sequences import get_device, get_dtype


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annarbm train", description="Train an annaRBM model.")
    return add_args_train(parser)


def _output_paths(args) -> dict:
    prefix = args.label if args.label is not None else "annaRBM"
    return {
        "log": os.path.join(args.output, f"{prefix}.log"),
        "checkpoint": os.path.join(args.output, f"{prefix}_checkpoints" if args.label is not None else "checkpoints"),
        "weights": os.path.join(args.output, f"{prefix}_weights.dat" if args.label is not None else "weights.dat"),
    }


def _prepare_output(paths: dict, resume: bool, overwrite: bool):
    """Creates the output folders. When starting a new training, existing outputs are deleted only if `overwrite`."""
    existing = [p for p in (paths["log"], paths["checkpoint"]) if os.path.exists(p)]
    if existing and not resume:
        if not overwrite:
            raise FileExistsError(f"{', '.join(existing)} already exist(s). Use --overwrite to replace, -p to resume the training, or change --label/--output.")
        for p in existing:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(paths["checkpoint"], exist_ok=True)


def _build_model(args, dataset: AnnaDataset):
    return get_rbm(
        visible_type="bernoulli" if dataset.is_binary else "potts",
        hidden_type=args.potential,
        visible_shape=dataset.L if dataset.is_binary else (dataset.L, dataset.q),
        hidden_shape=args.hidden,
        num_classes=dataset.get_num_classes(),
        continuous_label=args.continuous_labels,
    )


def learning_rate(update: int, lr: float, lr_final: Optional[float], num_updates: int) -> float:
    """Learning rate at a given update: constant, or decaying from lr to lr_final with a cosine schedule."""
    if lr_final is None:
        return lr
    progress = min(update / max(num_updates, 1), 1.0)
    return lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * progress))


def _load_checkpoint(args, dataset, device, dtype):
    """Loads the model and the checkpoint. Older checkpoints without configuration are rebuilt from the command line arguments."""
    try:
        return load_rbm(args.checkpoint, device=device, dtype=dtype)
    except ValueError:
        warnings.warn("The checkpoint does not contain the model configuration: the model is built from the command line arguments.")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        rbm = _build_model(args, dataset).to(device=device, dtype=dtype)
        rbm.load_state_dict(checkpoint["model_state_dict"])
        checkpoint["chains"] = rbm.decompress_chains(checkpoint["chains"])
        return rbm, checkpoint


def main(argv: Optional[List[str]] = None):
    args = create_parser().parse_args(argv)
    torch.set_float32_matmul_precision("high")

    print("\n" + "".join(["*"] * 10) + " Training annaRBM model " + "".join(["*"] * 10) + "\n")
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    standardize = not (args.no_standardize or args.no_centering)
    template = "{0:<30} {1:<50}"
    for name, value in [
        ("Input MSA:", args.data), ("Input annotations:", args.annotations), ("Output folder:", args.output),
        ("Number of hidden units:", args.hidden), ("Hidden potential:", args.potential), ("Learning rate:", args.lr),
        ("Final learning rate:", args.lr_final), ("Number of chains:", args.nchains), ("Batch size:", args.batch_size or args.nchains),
        ("Number of Gibbs steps:", args.gibbs_steps), ("Number of updates:", args.nupdates),
        ("Centered gradient:", not args.no_centering), ("Standardized gradient:", standardize),
        ("L1 / L2 / L1L2 regularization:", f"{args.l1} / {args.l2} / {args.l1l2}"), ("Label L1 / L2 regularization:", f"{args.l1_label} / {args.l2_label}"),
        ("Data initialization:", args.init_from_data), ("Continuous labels:", args.continuous_labels),
        ("Missing labels:", args.missing_labels), ("Random seed:", args.seed), ("Data type:", args.dtype),
    ]:
        print(template.format(name, str(value)))
    print("\n")

    for path in [args.data, args.annotations, args.checkpoint, args.weights]:
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Input file {path} not found.")
    torch.manual_seed(args.seed)

    print("Importing dataset...")
    dataset = AnnaDataset(
        path_data=args.data,
        path_ann=args.annotations,
        column_names=args.column_names,
        column_sequences=args.column_sequences,
        column_labels=args.column_labels,
        is_binary=args.is_binary,
        continuous_labels=args.continuous_labels,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        path_weights=args.weights,
        device=device,
        dtype=dtype,
    )
    print(f"Alphabet: {dataset.tokens}")
    print(f"Dataset imported successfully: M={len(dataset)}, L={dataset.L}, q={dataset.q}, Meff={dataset.get_effective_size():.1f}.")

    paths = _output_paths(args)
    resume = args.checkpoint is not None
    _prepare_output(paths, resume=resume, overwrite=args.overwrite)
    if args.weights is None and not args.no_reweighting:
        np.savetxt(paths["weights"], dataset.weights.cpu().numpy())
        print(f"Weights saved in {paths['weights']}")
    elif args.no_reweighting:
        print("All sequence weights set to 1.0")

    if args.pseudocount is None:
        args.pseudocount = 1.0 / dataset.get_effective_size()
        print(f"Pseudocount automatically set to 1/Meff = {args.pseudocount:.3g}.")
    batch_size = args.batch_size or args.nchains
    if batch_size > len(dataset):
        warnings.warn(f"The batch size ({batch_size}) is larger than the dataset size: it is set to {len(dataset)}.")
        batch_size = len(dataset)

    # Model, chains and optimizer
    if resume:
        rbm, checkpoint = _load_checkpoint(args, dataset, device, dtype)
        chains = checkpoint["chains"]
        upd = checkpoint["update"]
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        print(f"Model parameters and chains loaded from {args.checkpoint} (update {upd}).")
    else:
        rbm = _build_model(args, dataset).to(device=device, dtype=dtype)
        if args.init_from_data:
            rbm.init_params_from_data(visible=dataset.data_one_hot, label=dataset.labels, weights=dataset.weights, pseudo_count=args.pseudocount)
        chains = rbm.init_chains(
            num_samples=args.nchains,
            data=dataset.data_one_hot,
            label=dataset.labels,
            weights=dataset.weights,
            pseudo_count=args.pseudocount,
        )
        upd = 0
    optimizer = SGD(rbm.parameters(), lr=args.lr, maximize=True)
    if resume:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, KeyError):
            warnings.warn("The optimizer state of the checkpoint is not compatible and is not restored.")
    for p in rbm.parameters():
        p.grad = torch.zeros_like(p)
    metadata = {**dataset.metadata(), "args": {k: v for k, v in vars(args).items()}}

    gradient_kwargs = dict(
        pseudo_count=args.pseudocount,
        l1_strength=args.l1,
        l2_strength=args.l2,
        l1l2_strength=args.l1l2,
        l1_label_strength=args.l1_label,
        l2_label_strength=args.l2_label,
        standardize=standardize,
        centered=not args.no_centering,
        missing_labels=args.missing_labels,
    )

    # Fixed subset of the data used to monitor the training
    eval_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:5000].to(device)
    log_template = "{0:<10} {1:<12} {2:<12} {3:<14} {4:<12}\n"

    def log_metrics(f):
        pearson = two_point_pearson(dataset.data_one_hot, chains["visible"], dataset.weights)
        accuracy = float("nan") if args.continuous_labels else label_accuracy(rbm, dataset.data_one_hot[eval_idx], dataset.labels[eval_idx])
        lr = learning_rate(upd, args.lr, args.lr_final, args.nupdates)
        f.write(log_template.format(upd, f"{time.time() - start:.2f}", f"{lr:.3g}", f"{pearson:.3f}", f"{accuracy:.3f}"))
        f.flush()

    with open(paths["log"], "a" if resume else "w") as f:
        if resume:
            f.write(f"\nResumed from {args.checkpoint} at update {upd}\n")
        else:
            for name, value in [
                ("label:", args.label or "N/A"), ("input data:", args.data), ("alphabet:", dataset.tokens),
                ("# hiddens:", args.hidden), ("potential:", args.potential), ("nchains:", args.nchains),
                ("batch size:", batch_size), ("gibbs steps:", args.gibbs_steps), ("lr:", args.lr), ("lr final:", args.lr_final),
                ("max grad norm:", args.max_grad_norm), ("pseudo count:", args.pseudocount),
                ("centered:", not args.no_centering), ("standardized:", standardize), ("data init:", args.init_from_data),
                ("continuous labels:", args.continuous_labels), ("missing labels:", args.missing_labels),
                ("l1 strength:", args.l1), ("l2 strength:", args.l2), ("l1l2 strength:", args.l1l2),
                ("l1 label strength:", args.l1_label), ("l2 label strength:", args.l2_label), ("random seed:", args.seed),
            ]:
                f.write("{0:<20} {1:<10}\n".format(name, str(value)))
            f.write("\n")
            f.write(log_template.format("Update", "Time [s]", "lr", "Pearson Cij", "Label acc."))

        if not args.no_compile:
            rbm.compile_sampling()
            print("Sampling functions will be compiled at the first iteration.\n")

        batches = dataset.iterate_batches(batch_size, generator=torch.Generator().manual_seed(args.seed + upd))
        start = time.time()
        if upd == 0:
            save_checkpoint(rbm, chains, optimizer, upd, save_dir=paths["checkpoint"], metadata=metadata)
            log_metrics(f)

        rbm.train()
        with torch.no_grad(), tqdm(initial=upd, total=args.nupdates, colour="red", dynamic_ncols=True, ascii="-#") as pbar:
            while upd < args.nupdates:
                upd += 1
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(upd - 1, args.lr, args.lr_final, args.nupdates)
                optimizer.zero_grad(set_to_none=False)
                chains = rbm.forward(data_batch=next(batches), chains=chains, gibbs_steps=args.gibbs_steps, **gradient_kwargs)
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                pbar.update(1)

                if upd % args.save_every == 0 or upd == args.nupdates:
                    save_checkpoint(rbm, chains, optimizer, upd, save_dir=paths["checkpoint"], metadata=metadata)
                    log_metrics(f)


if __name__ == "__main__":
    main()
