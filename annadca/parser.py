import argparse
from pathlib import Path

def add_args_train(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("Training an annaRBM model.")
    dca_args.add_argument("-d", "--data",         type=Path,  required=True,        help="Filename of the dataset to be used for training the model.")
    dca_args.add_argument("-o", "--output",       type=Path,  default="annaRBM",    help="(Defaults to annaRBM). Path to the folder where to save the model.")
    dca_args.add_argument("-a", "--annotations",  type=Path,  required=True,        help="Path to the file containing the annotations of the sequences.")
    dca_args.add_argument("-H", "--hidden",       type=int,   default=100,          help="(Defaults to 100). Number of hidden units.")
    # Optional arguments
    dca_args.add_argument("-w", "--weights",      type=Path,  default=None,         help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    dca_args.add_argument("-p", "--path_params",  type=Path,  default=None,         help="(Defaults to None) Path to the file containing the model's parameters. Required for restoring the training.")
    dca_args.add_argument("-c", "--path_chains",  type=Path,  default=None,         help="(Defaults to None) Path to the fasta file containing the model's chains. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--gibbs_steps",        type=int,   default=10,           help="(Defaults to 10). Number of Alternating Gibbs steps for each gradient estimation.")
    dca_args.add_argument("--nchains",            type=int,   default=2000,         help="(Defaults to 2000). Number of Markov chains to run in parallel. It also corresponds to the batch size.")
    dca_args.add_argument("--nepochs",            type=int,   default=1000,         help="(Defaults to 1000). Maximum number of gradient updates allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=1e-8,         help="(Defaults to 1e-8). Pseudo count for the single and two-sites statistics. Acts as a regularization.")
    dca_args.add_argument("--eta",                type=float, default=1.0,          help="(Defaults to 1.0). Relative contribution of the label term.")
    dca_args.add_argument("--uncentered",         action="store_true",              help="(Defaults to False). If specified, the uncentered version of the gradient is used.")
    dca_args.add_argument("--init_from_profile",  action="store_true",              help="(Defaults to False). If specified, the biases are initialized using the independent-site model.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.", choices=["float32", "float64"])
    return parser