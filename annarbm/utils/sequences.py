from typing import Iterable
import numpy as np
import torch
from torch.nn.functional import one_hot


TOKENS_PROTEIN = "-ACDEFGHIKLMNPQRSTVWY"
TOKENS_RNA = "-ACGU"
TOKENS_DNA = "-ACGT"


def get_tokens(alphabet: str) -> str:
    """Returns the tokens of a built-in alphabet ('protein', 'rna', 'dna'), or the alphabet itself if it is a custom string of tokens.

    Args:
        alphabet (str): Name of the alphabet or string of tokens.

    Returns:
        str: String of tokens.
    """
    if not isinstance(alphabet, str):
        raise TypeError("Argument 'alphabet' must be of type str")
    builtin = {"protein": TOKENS_PROTEIN, "rna": TOKENS_RNA, "dna": TOKENS_DNA}
    tokens = builtin.get(alphabet, alphabet)
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"The alphabet '{tokens}' contains repeated tokens.")
    return tokens


def encode_sequence(sequence: str | Iterable[str], tokens: str) -> np.ndarray:
    """Encodes a sequence, or a list of aligned sequences, into integers.

    Args:
        sequence (str | Iterable[str]): Sequence or iterable of sequences of the same length.
        tokens (str): Alphabet used for the encoding.

    Returns:
        np.ndarray: Array of shape (L,) or (num_sequences, L).
    """
    letter_map = {letter: n for n, letter in enumerate(tokens)}
    if isinstance(sequence, str):
        return np.array([letter_map[letter] for letter in sequence], dtype=np.int64)
    encoded = [[letter_map[letter] for letter in seq] for seq in sequence]
    return np.array(encoded, dtype=np.int64).reshape(len(encoded), -1)


def decode_sequence(sequence: np.ndarray | torch.Tensor, tokens: str) -> str | list[str]:
    """Decodes integer-encoded or one-hot encoded sequences into strings.

    Args:
        sequence (np.ndarray | torch.Tensor): Array of shape (L,), (num_sequences, L) or one-hot (num_sequences, L, q).
        tokens (str): Alphabet used for the encoding.

    Returns:
        str | list[str]: Decoded sequence or list of sequences.
    """
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.cpu().numpy()
    sequence = np.asarray(sequence)
    if sequence.ndim == 3:
        sequence = sequence.argmax(axis=-1)
    if sequence.ndim == 1:
        return "".join(tokens[int(x)] for x in sequence)
    return ["".join(tokens[int(x)] for x in seq) for seq in sequence]


@torch.no_grad()
def compute_weights(
    data: np.ndarray | torch.Tensor,
    th: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Computes the weight of each sequence as 1 / n, where n is the number of sequences in the dataset (itself included)
    whose sequence identity with it is larger than `th`.

    Args:
        data (np.ndarray | torch.Tensor): Integer-encoded (num_sequences, L) or one-hot encoded (num_sequences, L, q) sequences.
        th (float, optional): Sequence identity threshold. Defaults to 0.8.
        device (torch.device, optional): Device used for the computation. Defaults to cpu.
        dtype (torch.dtype, optional): Data type of the weights. Defaults to torch.float32.
        chunk_size (int, optional): Number of sequences processed at once. Defaults to 1024.

    Returns:
        torch.Tensor: Weights of shape (num_sequences,).
    """
    data = torch.as_tensor(data, device=device)
    if data.dim() == 3:
        data = data.argmax(dim=-1)
    if data.dim() != 2:
        raise ValueError("'data' must be either a (num_sequences, L) or a (num_sequences, L, q) (one-hot encoded) array.")
    data = data.long()
    L = data.shape[1]
    # Sequence identity as a scalar product of one-hot encodings
    x = one_hot(data, num_classes=int(data.max()) + 1).flatten(1).to(torch.float32)
    counts = torch.cat([
        ((x[i:i + chunk_size] @ x.T) / L > th).sum(dim=1)
        for i in range(0, len(x), chunk_size)
    ])
    return (1.0 / counts).to(dtype)


def get_device(device: str, message: bool = True) -> torch.device:
    """Returns the torch device. 'auto' selects CUDA, then MPS, then CPU. Falls back to CPU if CUDA is not available.

    Args:
        device (str): 'auto', 'cpu', 'cuda', 'cuda:<index>' or 'mps'.
        message (bool, optional): Whether to print the selected device. Defaults to True.

    Returns:
        torch.device: Device.
    """
    device = str(device).lower()
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if device.startswith("mps") and torch.backends.mps.is_available():
        if message:
            print("Running on M chip GPU, Metal Performance Shaders (MPS)")
        return torch.device(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        if message:
            print(f"Running on {torch.cuda.get_device_name(torch.device(device))}")
        return torch.device(device)
    if message:
        print("Running on CPU")
    return torch.device("cpu")


def get_dtype(dtype: str) -> torch.dtype:
    """Returns the torch data type from its name ('float32' or 'float64')."""
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    if dtype not in dtypes:
        raise ValueError(f"Data type {dtype} not supported. Choose among {list(dtypes)}.")
    return dtypes[dtype]
