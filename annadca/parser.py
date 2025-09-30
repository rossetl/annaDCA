import argparse

def add_args_annadca(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("Training an annaRBM model.")
    dca_args.add_argument("-d", "--data",         type=str,  required=True,         help="Filename of the dataset to be used for training the model. Can be a fasta file or a csv file containing the sequences and their annotations.")
    dca_args.add_argument("-o", "--output",       type=str,  default="annaRBM",     help="(Defaults to ./annaRBM). Path to the folder where to save the model.")
    dca_args.add_argument("-a", "--annotations",  type=str,  default=None,          help="Path to the file containing the annotations of the sequences.")
    dca_args.add_argument("-H", "--hidden",       type=int,   default=100,          help="(Defaults to 100). Number of hidden units.")
    dca_args.add_argument("--potential",          type=str,   default="bernoulli",  help="(Defaults to bernoulli). Type of the hidden potential to be used. Choose among ['bernoulli', 'gaussian', 'relu'].")
    dca_args.add_argument("--column_names",       type=str,   default="name",       help="(Defaults to name). Name of the column containing the names of the sequences in the annotations file.")
    dca_args.add_argument("--column_sequences",   type=str,   default="sequence",   help="(Defaults to sequence). Name of the column containing the sequences in the annotations file.")
    dca_args.add_argument("--column_labels",      type=str,   default="label",      help="(Defaults to label). Name of the column containing the labels in the annotations file.")
    dca_args.add_argument("-p", "--checkpoint",   type=str,  default=None,          help="(Defaults to None) Path to the file containing the model's checkpoint. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--gibbs_steps",        type=int,   default=10,           help="(Defaults to 10). Number of Alternating Gibbs steps for each gradient estimation.")
    dca_args.add_argument("--nchains",            type=int,   default=5000,         help="(Defaults to 5000). Number of Markov chains to run in parallel. It also corresponds to the batch size.")
    dca_args.add_argument("--nepochs",            type=int,   default=10000,        help="(Defaults to 10000). Maximum number of gradient updates allowed.")
    dca_args.add_argument("--pseudocount",        type=float, default=1e-8,         help="(Defaults to 1e-8). Pseudo count for the single and two-sites statistics. Acts as a regularization.")
    dca_args.add_argument("--l1",                 type=float, default=0.0,          help="(Defaults to 0.0). L1 regularization strength.")
    dca_args.add_argument("--l2",                 type=float, default=0.0,          help="(Defaults to 0.0). L2 regularization strength.")
    dca_args.add_argument("--l1l2",               type=float, default=0.0,          help="(Defaults to 0.0). L1+L2 regularization strength.")
    dca_args.add_argument("--uncentered",         action="store_true",              help="(Defaults to False). If specified, the uncentered version of the gradient is used.")
    dca_args.add_argument("--init_from_profile",  action="store_true",              help="(Defaults to False). If specified, the biases are initialized using the independent-site model.")
    dca_args.add_argument("--is_binary",          action="store_true",              help="(Defaults to False). If specified, the sequences are assumed to be binary (0/1) and the model is trained accordingly.")
    dca_args.add_argument("--save_every",         type=int,   default=1000,         help="(Defaults to 1000). Frequency (in number of updates) at which the model is saved.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.", choices=["float32", "float64"])
    
    return parser

def add_args_reweighting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    reweighting_args = parser.add_argument_group("Sequence reweighting arguments")
    reweighting_args.add_argument("-w", "--weights",      type=str,  default=None,  help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    reweighting_args.add_argument("--no_reweighting",     action="store_true",       help="(Defaults to False). If provided, the reweighting of the sequences is not performed.")
    reweighting_args.add_argument("--clustering_seqid",   type=float, default=0.8,   help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")
    
    return parser

def add_args_train(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_annadca(parser)
    parser = add_args_reweighting(parser)
    
    return parser