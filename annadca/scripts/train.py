import argparse
import os
import shutil
import numpy as np
from tqdm import tqdm
from itertools import cycle
import time
import warnings
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from adabmDCA.utils import get_device, get_dtype
from annadca.parser import add_args_train
from annadca.dataset import annaDataset
from annadca.rbm import get_rbm, save_checkpoint
from annadca.utils.stats import get_mean

torch.set_float32_matmul_precision('high')

# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Train an annaRBM model.')
    parser = add_args_train(parser) 
    return parser


if __name__ == '__main__':
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training annaRBM model " + "".join(["*"] * 10) + "\n")
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Input annotations:", str(args.annotations)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Number of hidden units:", args.hidden))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Minibatch size:", args.nchains))
    print(template.format("Number of chains:", args.nchains))
    print(template.format("Number of Gibbs steps:", args.gibbs_steps))
    print(template.format("Number of epochs:", args.nepochs))
    print(template.format("Standardized gradient:", str(not args.no_standardize)))
    print(template.format("L1 regularization:", args.l1))
    print(template.format("L2 regularization:", args.l2))
    print(template.format("Profile initialization:", args.init_from_profile))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Data type:", args.dtype))
    print("\n")

    # Check that input files exist
    for path in [args.data, args.annotations, args.checkpoint]:
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
    data = dataset.data_one_hot
    tokens = dataset.tokens
    print(f"Alphabet: {dataset.tokens}")
    print(f"Dataset imported successfully: M={len(dataset)}, L={dataset.L}, q={dataset.q}.")
            
    # Create the folder where to save the model
    folder = args.output
    os.makedirs(folder, exist_ok=True)
    
    if args.label is not None:
        file_paths = {
            "log": os.path.join(folder, f"{args.label}.log"),
            "checkpoint": os.path.join(folder, f"{args.label}_checkpoints"),
        }
    else:
        file_paths = {
            "log": os.path.join(folder, f"annaRBM.log"),
            "checkpoint": os.path.join(folder, f"checkpoints"),
        }
    
    # Check if files already exist and delete them before creating new ones
    if args.checkpoint is None:
        log_path = file_paths["log"]
        if os.path.exists(log_path):
            confirm = input(f"File {log_path} already exists. Do you want to delete it? (y/n): ")
            if confirm.lower() == "y":
                os.remove(log_path)  # Use os.remove() for files
        
        # Handle checkpoint directory
        checkpoint_path = file_paths["checkpoint"]
        if os.path.exists(checkpoint_path):
            confirm = input(f"Directory {checkpoint_path} already exists. Do you want to delete it? (y/n): ")
            if confirm.lower() == "y":
                shutil.rmtree(checkpoint_path)
    
    # Create subfolder for checkpoints
    os.makedirs(file_paths["checkpoint"], exist_ok=True)

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
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")
        
    # Initialize the model and the chains
    num_visibles = dataset.get_num_residues()
    num_hiddens = args.hidden
    num_classes = dataset.get_num_classes()
    num_states = dataset.get_num_states()
    
    if dataset.is_binary:
        rbm = get_rbm(
            visible_type="bernoulli",
            hidden_type=args.potential,
            visible_shape=num_visibles,
            hidden_shape=num_hiddens,
            num_classes=num_classes,
        )
        rbm.to(device=device, dtype=dtype)
    elif not dataset.is_binary:
        rbm = get_rbm(
            visible_type="potts",
            hidden_type=args.potential,
            visible_shape=(num_visibles, num_states),
            hidden_shape=num_hiddens,
            num_classes=num_classes,
        )
        rbm.to(device=device, dtype=dtype)
        
    frequences_visible = get_mean(
        x=data,
        weights=dataset.weights,
        pseudo_count=args.pseudocount,
    )
    frequences_labels = get_mean(
        x=dataset.labels_one_hot,
        weights=dataset.weights,
        pseudo_count=args.pseudocount,
    )

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        rbm.load_state_dict(checkpoint['model_state_dict'])
        chains = checkpoint["chains"]
        # Select the optimizer
        optimizer = SGD(rbm.parameters(), lr=args.lr, maximize=True)
        if args.checkpoint is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        upd = checkpoint['update']
        print("Model parameters and chains loaded from", args.checkpoint)
        
    else:
        if args.init_from_profile:            
            rbm.init_from_frequencies(
                frequencies_visible=frequences_visible,
                frequencies_label=frequences_labels,
            )
        if args.nchains >= dataset.__len__():
            args.nchains = dataset.__len__()
            warnings.warn("The number of chains is larger than the dataset size. The number of chains is set to the dataset size.")
        chains = rbm.init_chains(num_samples=args.nchains, frequencies=frequences_visible)
        # Select the optimizer
        optimizer = SGD(rbm.parameters(), lr=args.lr, maximize=True)
        for key, value in rbm.named_parameters():
            value.grad = torch.zeros_like(value)
        upd = 0
        
    print("\n")
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
        f.write(template.format("lr:", args.lr))
        f.write(template.format("pseudo count:", args.pseudocount))
        f.write(template.format("standardized:", str(not args.no_standardize)))
        f.write(template.format("profile init:", args.init_from_profile))
        f.write(template.format("l1 strength:", args.l1))
        f.write(template.format("l2 strength:", args.l2))
        f.write(template.format("random seed:", args.seed))
        f.write("\n")
        template = "{0:<10} {1:<10}\n"
        f.write(template.format("Epoch", "Time [s]"))

    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.nchains,
        shuffle=True,
        drop_last=True,
    )
    
    # Allows to iterate indefinitely on the dataloader without worrying on the epochs
    dataloader = cycle(dataloader)
    
    if torch.__version__ >= "2.0.0":
        print("Compiling the model...")
        rbm = torch.compile(rbm)
        print("Model compiled successfully.")
    print("\n")

    # Train the model
    start = time.time()
    pbar = tqdm(initial=upd, total=args.nepochs, colour="red", dynamic_ncols=True, ascii="-#")
    if upd == 0:
        save_checkpoint(rbm, chains, optimizer, upd, save_dir=file_paths["checkpoint"])
    
    rbm.train()
    with torch.no_grad():
        while upd < args.nepochs:
            upd += 1
            if upd % 10 == 0:
                pbar.update(10)

            # Get the next batch
            batch = next(dataloader)
            optimizer.zero_grad(set_to_none=False)
            chains = rbm.forward(
                data_batch=batch,
                chains=chains,
                gibbs_steps=args.gibbs_steps,
                pseudo_count=args.pseudocount,
                l1_strength=args.l1,
                l2_strength=args.l2,
                l1l2_strength=args.l1l2,
                standardize=not args.no_standardize,
            )
            
            # normalize the gradients
            # torch.nn.utils.clip_grad_norm_(rbm.parameters(), max_norm=5.0)

            # Update the parameters
            optimizer.step()

            if upd % args.save_every == 0:
                save_checkpoint(rbm, chains, optimizer, upd, save_dir=file_paths["checkpoint"])
                with open(file_paths["log"], "a") as f:
                    f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))
        pbar.close()
        
    # Save the final model if not already saved
    if upd % args.save_every != 0:
        save_checkpoint(rbm, chains, optimizer, upd, save_dir=file_paths["checkpoint"])
        with open(file_paths["log"], "a") as f:
            f.write(template.format(f"{upd}", f"{time.time() - start:.2f}"))