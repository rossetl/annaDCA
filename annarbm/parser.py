import argparse


def add_args_model(parser : argparse.ArgumentParser) -> argparse.ArgumentParser:
    dca_args = parser.add_argument_group("Training an annaRBM model.")
    dca_args.add_argument("-d", "--data",         type=str,  required=True,         help="Filename of the dataset to be used for training the model. Can be a fasta file or a csv file containing the sequences and their annotations.")
    dca_args.add_argument("-o", "--output",       type=str,  default="annaRBM",     help="(Defaults to ./annaRBM). Path to the folder where to save the model.")
    dca_args.add_argument("-a", "--annotations",  type=str,  default=None,          help="Path to the file containing the annotations of the sequences.")
    dca_args.add_argument("-H", "--hidden",       type=int,   default=100,          help="(Defaults to 100). Number of hidden units.")
    dca_args.add_argument("--potential",          type=str,   default="bernoulli",  help="(Defaults to bernoulli). Type of the hidden potential to be used.", choices=["bernoulli", "gaussian", "relu"])
    dca_args.add_argument("--column_names",       type=str,   default="name",       help="(Defaults to name). Name of the column containing the names of the sequences in the annotations file.")
    dca_args.add_argument("--column_sequences",   type=str,   default="sequence",   help="(Defaults to sequence). Name of the column containing the sequences in the annotations file.")
    dca_args.add_argument("--column_labels",      type=str,   default="label",      help="(Defaults to label). Name of the column containing the labels in the annotations file.")
    dca_args.add_argument("-p", "--checkpoint",   type=str,  default=None,          help="(Defaults to None) Path to the file containing the model's checkpoint. Required for restoring the training.")
    dca_args.add_argument("-l", "--label",        type=str,   default=None,         help="(Defaults to None). If provoded, adds a label to the output files inside the output folder.")
    dca_args.add_argument("--alphabet",           type=str,   default="protein",    help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    dca_args.add_argument("--lr",                 type=float, default=0.01,         help="(Defaults to 0.01). Learning rate.")
    dca_args.add_argument("--lr_final",           type=float, default=None,         help="(Defaults to None). If provided, the learning rate decays from lr to lr_final with a cosine schedule over the training. Otherwise, it is constant.")
    dca_args.add_argument("--max_grad_norm",      type=float, default=None,         help="(Defaults to None). If provided, the norm of the gradient of all the parameters is clipped to this value.")
    dca_args.add_argument("--gibbs_steps",        type=int,   default=10,           help="(Defaults to 10). Number of Alternating Gibbs steps for each gradient estimation.")
    dca_args.add_argument("--nchains",            type=int,   default=5000,         help="(Defaults to 5000). Number of Markov chains to run in parallel.")
    dca_args.add_argument("--batch_size",         type=int,   default=None,         help="(Defaults to nchains). Number of data sequences used for each gradient estimation.")
    dca_args.add_argument("--nupdates", "--nepochs", type=int, default=10000, dest="nupdates", help="(Defaults to 10000). Number of gradient updates. '--nepochs' is a deprecated alias.")
    dca_args.add_argument("--pseudocount",        type=float, default=None,         help="(Defaults to 1/Meff). Pseudo count for the statistics of the discrete variables. Acts as a regularization.")
    dca_args.add_argument("--l1",                 type=float, default=0.0,          help="(Defaults to 0.0). L1 regularization strength of the weight matrix.")
    dca_args.add_argument("--l2",                 type=float, default=0.0,          help="(Defaults to 0.0). L2 regularization strength of the weight matrix.")
    dca_args.add_argument("--l1l2",               type=float, default=0.0,          help="(Defaults to 0.0). L1^2 regularization strength of the weight matrix.")
    dca_args.add_argument("--l1_label",           type=float, default=0.0,          help="(Defaults to 0.0). L1 regularization strength of the label matrix.")
    dca_args.add_argument("--l2_label",           type=float, default=0.0,          help="(Defaults to 0.0). L2 regularization strength of the label matrix.")
    dca_args.add_argument("--no_standardize",     action="store_true",              help="(Defaults to False). If specified, the gradient is only centered and not rescaled by the standard deviation of the units.")
    dca_args.add_argument("--no_centering",       action="store_true",              help="(Defaults to False). If specified, the plain (neither centered nor standardized) gradient is used.")
    dca_args.add_argument("--continuous_labels",  action="store_true",              help="(Defaults to False). If specified, the labels are treated as continuous variables.")
    dca_args.add_argument("--missing_labels",     type=str,   default="posterior",  help="(Defaults to posterior). How to treat the sequences without annotation (categorical labels only). 'posterior': the missing label is summed over with its posterior p(label | sequence); 'zero': the label is set to the all-zero vector.", choices=["posterior", "zero"])
    dca_args.add_argument("--init_from_data",     action="store_true",              help="(Defaults to False). If specified, the visible layer parameters are initialized using the data statistics.")
    dca_args.add_argument("--is_binary",          action="store_true",              help="(Defaults to False). If specified, the sequences are assumed to be binary (0/1) and the model is trained accordingly.")
    dca_args.add_argument("--save_every",         type=int,   default=1000,         help="(Defaults to 1000). Frequency (in number of updates) at which the model is saved and the training metrics are logged.")
    dca_args.add_argument("--overwrite",          action="store_true",              help="(Defaults to False). If specified, an existing log file and checkpoint folder with the same name are deleted. Otherwise, the training stops if they exist.")
    dca_args.add_argument("--seed",               type=int,   default=0,            help="(Defaults to 0). Seed for the random number generator.")
    dca_args.add_argument("--no_compile",         action="store_true",              help="(Defaults to False). If specified, the sampling functions are not compiled with torch.compile.")
    dca_args.add_argument("--device",             type=str,   default="cuda",       help="(Defaults to cuda). Device to be used: 'cuda', 'mps', 'cpu' or 'auto'. Falls back to cpu if cuda is not available.")
    dca_args.add_argument("--dtype",              type=str,   default="float32",    help="(Defaults to float32). Data type to be used.", choices=["float32", "float64"])

    return parser

def add_args_reweighting(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    reweighting_args = parser.add_argument_group("Sequence reweighting arguments")
    reweighting_args.add_argument("-w", "--weights",      type=str,  default=None,  help="(Defaults to None). Path to the file containing the weights of the sequences. If None, the weights are computed automatically.")
    reweighting_args.add_argument("--no_reweighting",     action="store_true",       help="(Defaults to False). If provided, the reweighting of the sequences is not performed.")
    reweighting_args.add_argument("--clustering_seqid",   type=float, default=0.8,   help="(Defaults to 0.8). Sequence Identity threshold for clustering. Used only if 'weights' is not provided.")

    return parser

def add_args_train(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = add_args_model(parser)
    parser = add_args_reweighting(parser)

    return parser


def add_args_sample(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    sample_args = parser.add_argument_group("Sampling from an annaRBM model.")
    sample_args.add_argument("-p", "--checkpoint",  type=str,   required=True,       help="Path to the model checkpoint.")
    sample_args.add_argument("-o", "--output",      type=str,   required=True,       help="Path to the output fasta file.")
    sample_args.add_argument("-n", "--num_samples", type=int,   default=1000,        help="(Defaults to 1000). Number of sequences to generate.")
    sample_args.add_argument("--gibbs_steps",       type=int,   default=1000,        help="(Defaults to 1000). Number of Alternating Gibbs steps.")
    sample_args.add_argument("--beta",              type=float, default=1.0,         help="(Defaults to 1.0). Inverse temperature.")
    sample_args.add_argument("--target",            type=str,   default=None,        help="(Defaults to None). If provided, the sequences are sampled conditioned on this label (a class name, or a value for continuous labels). Otherwise, labels are sampled jointly with the sequences.")
    sample_args.add_argument("--seed",              type=int,   default=0,           help="(Defaults to 0). Seed for the random number generator.")
    sample_args.add_argument("--device",            type=str,   default="cuda",      help="(Defaults to cuda). Device to be used: 'cuda', 'mps', 'cpu' or 'auto'.")
    sample_args.add_argument("--dtype",             type=str,   default="float32",   help="(Defaults to float32). Data type to be used.", choices=["float32", "float64"])
    return parser
