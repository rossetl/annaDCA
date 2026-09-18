import argparse
import json
import os
import time
import warnings

import h5py
import numpy as np
import torch
from adabmDCA.stats import get_freq_single_point as get_freq_single_point_cat
from adabmDCA.utils import get_device, get_dtype
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from annadca import annaRBMbin, annaRBMcat
from annadca.dataset import annaDataset
from annadca.io import _load_model_from_ptt
from annadca.parser import add_args_train
from annadca.rbm.binary.stats import get_freq_single_point as get_freq_single_point_bin
from annadca.train import pcd


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description="Train an annaRBM model.")
    parser = add_args_train(parser)

    return parser


def main(argv: list[str] | None = None) -> None:

    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args(argv)
    for name in ("hidden", "nchains", "nepochs", "gibbs_steps"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    print("\n" + "".join(["*"] * 10) + " Training annaRBM model " + "".join(["*"] * 10) + "\n")
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    backend = "triton" if args.triton else "torch"
    if args.triton:
        if device.type != "cuda" or dtype != torch.float32:
            parser.error("Triton training requires float32 CUDA; use --no-triton for this device/dtype")
        if not torch.cuda.is_available():
            parser.error("Triton training requires an available CUDA device; use --no-triton --device cpu")
        try:
            from annadca.rbm import triton_sampling  # noqa: F401
        except ImportError:
            parser.error("Triton is not installed; install annadca[triton] or use --no-triton")
    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Input annotations:", str(args.annotations)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Number of hidden units:", args.hidden))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Minibatch size:", args.nchains))
    print(template.format("Number of chains:", args.nchains))
    print(template.format("Number of Gibbs steps:", args.gibbs_steps))
    print(template.format("Sampling backend:", backend))
    print(template.format("Number of epochs:", args.nepochs))
    print(template.format("Centered gradient:", not args.uncentered))
    print(template.format("L1 regularization:", args.l1))
    print(template.format("L2 regularization:", args.l2))
    print(template.format("Profile initialization:", args.init_from_profile))
    print(template.format("Labels contribution:", args.eta))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Data type:", args.dtype))
    print("\n")

    # Check that input files exist
    for path in [
        args.data,
        args.annotations,
        args.path_params,
        args.path_chains,
        args.path_params_ptt,
    ]:
        if path is not None and not os.path.exists(path):
            raise FileNotFoundError(f"Input file {path} not found.")

    # Import data
    print("Importing dataset...")
    dataset = annaDataset(
        path_data=args.data,
        path_ann=args.annotations,
        column_names=args.column_names,
        column_sequences=args.column_sequences,
        column_labels=args.column_labels,
        is_binary=args.is_binary,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        path_weights=args.weights,
        device=device,
        dtype=dtype,
    )
    if args.path_params is not None:
        with h5py.File(args.path_params, "r") as archive:
            if "label_vocabulary" in archive.attrs and json.loads(
                archive.attrs["label_vocabulary"]
            ) != list(dataset.label_to_idx):
                raise ValueError("Resuming requires the original training label vocabulary.")
            if "alphabet" in archive.attrs and archive.attrs["alphabet"] != dataset.tokens:
                raise ValueError("Resuming requires the original training alphabet.")
    tokens = dataset.tokens
    print(f"Alphabet: {dataset.tokens}")
    print(f"Dataset imported successfully: M={len(dataset)}, L={dataset.L}, q={dataset.q}.")

    # Create the folder where to save the model
    folder = args.output
    os.makedirs(folder, exist_ok=True)

    if args.label is not None:
        file_paths = {
            "log": os.path.join(folder, f"{args.label}.log"),
            "params": os.path.join(folder, f"{args.label}_params.h5"),
            "chains": os.path.join(folder, f"{args.label}_chains.fasta"),
        }

    else:
        file_paths = {
            "log": os.path.join(folder, "annaRBM.log"),
            "params": os.path.join(folder, "params.h5"),
            "chains": os.path.join(folder, "chains.fasta"),
        }

    # Check if the files in file_paths already exist. If so, delete them
    if args.path_params is None and args.path_params_ptt is None:
        for path in file_paths.values():
            if os.path.exists(path):
                # Ask for confirmation before deleting
                confirm = input(f"File {path} already exists. Do you want to delete it? (y/n): ")
                if confirm.lower() != "y":
                    raise FileExistsError(f"Refusing to overwrite {path}")
                os.remove(path)

    # Save the weights if not already provided
    if args.weights is None and not args.no_reweighting:
        if args.label is not None:
            path_weights = os.path.join(folder, f"{args.label}_weights.dat")
        else:
            path_weights = os.path.join(folder, "weights.dat")
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"Weights saved in {path_weights}")
    elif args.no_reweighting:
        print("All sequence weights set to 1.0")

    # Set the random seed
    torch.manual_seed(args.seed)

    # Shuffle the dataset
    dataset.shuffle()

    if args.pseudocount is None:
        args.pseudocount = 1.0 / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")

    # Initialize the model and the chains
    num_visibles = dataset.get_num_residues()
    num_hiddens = args.hidden
    num_labels = dataset.get_num_classes()
    num_states = dataset.get_num_states()
    if dataset.is_binary:
        rbm = annaRBMbin()
        data = dataset.data_one_hot
        frequences_visible = get_freq_single_point_bin(
            data=data,
            weights=dataset.weights,
            pseudo_count=args.pseudocount,
        )
    elif not dataset.is_binary:
        rbm = annaRBMcat()
        data = dataset.data_one_hot
        frequences_visible = get_freq_single_point_cat(
            data=data,
            weights=dataset.weights,
            pseudo_count=args.pseudocount,
        )
    frequency_labels = get_freq_single_point_bin(
        data=dataset.labels_one_hot,
        weights=dataset.weights,
        pseudo_count=args.pseudocount,
    )

    if args.path_params_ptt is not None:
        if args.path_params is not None or args.path_chains is not None:
            raise ValueError(
                "Use either --path_params_ptt or --path_params/--path_chains, not both."
            )
        upd, params, chains = _load_model_from_ptt(
            filename=args.path_params_ptt,
            index=None,
            device=device,
            dtype=dtype,
            num_labels=num_labels,
            label_frequencies=frequency_labels.squeeze(0)
            if frequency_labels.ndim > 1
            else frequency_labels,
        )
        rbm.params = params
        rbm.device = device
        rbm.dtype = dtype

        if dataset.is_binary and params["weight_matrix"].dim() != 2:
            raise ValueError("Checkpoint mismatch: binary dataset expects a 2D weight_matrix.")
        if not dataset.is_binary and params["weight_matrix"].dim() != 3:
            raise ValueError("Checkpoint mismatch: categorical dataset expects a 3D weight_matrix.")

        if "label" not in chains:
            label_probs = frequency_labels.squeeze(0)
            label_probs = label_probs / label_probs.sum()
            label_idx = torch.multinomial(label_probs, len(chains["visible"]), replacement=True)
            chains["label"] = torch.nn.functional.one_hot(
                label_idx,
                num_classes=num_labels,
            ).to(device=device, dtype=dtype)

        if "hidden" not in chains:
            chains.update(
                rbm.sample_hiddens(
                    visible=chains["visible"],
                    label=chains["label"],
                    beta=1.0,
                )
            )

        args.nchains = len(chains["visible"])
        print(f"Model parameters and chains at update {upd} loaded from", args.path_params_ptt)
    elif args.path_params is not None:
        upd = rbm.load(
            filename=args.path_params,
            device=device,
            dtype=dtype,
            set_rng_state=True,
        )
        print(f"Model parameters at update {upd} loaded from", args.path_params)
    else:
        if args.init_from_profile:
            init_frequences_visible = frequences_visible
            init_frequences_labels = frequency_labels
        else:
            init_frequences_visible = None
            init_frequences_labels = None

        rbm.init_parameters(
            num_visibles=num_visibles,
            num_hiddens=num_hiddens,
            num_labels=num_labels,
            num_states=num_states,
            frequencies_visibles=init_frequences_visible,
            frequencies_labels=init_frequences_labels,
            std_init=1e-4,
            device=device,
            dtype=dtype,
        )
        upd = 0

    if args.path_chains is not None:
        chains = rbm.load_chains(
            filename=args.path_chains,
            device=device,
            dtype=dtype,
            alphabet=tokens,
        )
        print("Chains loaded from", args.path_chains)
    elif args.path_params_ptt is None:
        if args.nchains >= dataset.__len__():
            args.nchains = dataset.__len__()
            warnings.warn(
                "The number of chains is larger than the dataset size. The number of chains is set to the dataset size."
            )
        chains = rbm.init_chains(num_samples=args.nchains, use_profile=True)

    print("\n")
    # Save the hyperparameters of the model
    template = "{0:<20} {1:<10}\n"
    with open(file_paths["log"], "w") as f:
        if args.label is not None:
            f.write(template.format("label:", args.label))
        else:
            f.write(template.format("label:", "N/A"))

        f.write(template.format("input data:", str(args.data)))
        f.write(template.format("alphabet:", dataset.tokens))
        f.write(template.format("# hiddens:", args.hidden))
        f.write(template.format("nchains:", args.nchains))
        f.write(template.format("minibatch size:", args.nchains))
        f.write(template.format("gibbs steps:", args.gibbs_steps))
        f.write(template.format("sampling backend:", backend))
        f.write(template.format("lr:", args.lr))
        f.write(template.format("pseudo count:", args.pseudocount))
        f.write(template.format("centered:", not args.uncentered))
        f.write(template.format("profile init:", args.init_from_profile))
        f.write(template.format("l1 strength:", args.l1))
        f.write(template.format("l2 strength:", args.l2))
        f.write(template.format("eta:", args.eta))
        f.write(template.format("marginalize missing labels:", args.marginalize_missing_labels))
        f.write(template.format("random seed:", args.seed))
        f.write("\n")
        template = "{0:<10} {1:<10}\n"
        f.write(template.format("Epoch", "Time [s]"))

    # Initialize gradients for the parameters
    for key, value in rbm.params.items():
        value.grad = torch.zeros_like(value)

    # Select the optimizer
    optimizer = SGD(rbm.params.values(), lr=args.lr, maximize=True)

    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.nchains,
        shuffle=True,
        drop_last=True,
    )

    # Allows to iterate indefinitely on the dataloader without worrying on the epochs
    def batches():
        while True:
            yield from dataloader

    batch_iterator = batches()

    # Train the model
    start = time.time()
    pbar = tqdm(initial=upd, total=args.nepochs, colour="red", dynamic_ncols=True, ascii="-#")
    if upd == 0:
        rbm.save(filename=file_paths["params"], num_updates=upd)
    with torch.no_grad():
        while upd < args.nepochs:
            upd += 1
            if upd % 10 == 0:
                pbar.update(10)

            # Get the next batch
            batch = next(batch_iterator)
            optimizer.zero_grad(set_to_none=False)
            chains = pcd(
                rbm=rbm,
                data_batch=batch,
                chains=chains,
                gibbs_steps=args.gibbs_steps,
                pseudo_count=args.pseudocount,
                centered=(not args.uncentered),
                lambda_l1=args.l1,
                lambda_l2=args.l2,
                eta=args.eta,
                marginalize_labels=args.marginalize_missing_labels,
                backend=backend,
            )

            # Update the parameters
            optimizer.step()

            # Set the gauge of the weights
            rbm.zerosum_gauge()

            if upd % 1000 == 0:
                rbm.save(
                    filename=file_paths["params"],
                    num_updates=upd,
                )
                rbm.save_chains(
                    filename=file_paths["chains"],
                    visible=chains["visible"],
                    label=chains["label"],
                    alphabet=dataset.tokens,
                )
                with open(file_paths["log"], "a") as f:
                    f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))
        pbar.close()

        # Always leave a usable final checkpoint, including resumed runs in a new folder.
        rbm.save(filename=file_paths["params"], num_updates=upd)
        rbm.save_chains(
            filename=file_paths["chains"],
            visible=chains["visible"],
            label=chains["label"],
            alphabet=dataset.tokens,
        )
        with open(file_paths["log"], "a") as f:
            f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))

    with h5py.File(file_paths["params"], "a") as archive:
        archive.attrs["label_vocabulary"] = json.dumps(list(dataset.label_to_idx))
        archive.attrs["alphabet"] = dataset.tokens


if __name__ == "__main__":
    main()
