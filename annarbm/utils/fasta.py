from typing import List, Tuple, Iterable


def read_fasta(filepath: str) -> Tuple[List[str], List[str]]:
    """Reads a fasta file.

    Args:
        filepath (str): Path to the fasta file.

    Returns:
        Tuple[List[str], List[str]]: Headers (without the leading '>') and sequences.
    """
    headers, sequences = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                headers.append(line[1:].strip())
                sequences.append("")
            else:
                if len(headers) == 0:
                    raise ValueError(f"Invalid fasta file {filepath}: sequence found before any header.")
                sequences[-1] += line
    return headers, sequences


def write_fasta(
    filepath: str,
    headers: Iterable[str],
    sequences: Iterable[str],
):
    """Writes headers and sequences to a fasta file.

    Args:
        filepath (str): Path to the output fasta file.
        headers (Iterable[str]): Sequence headers (without the leading '>').
        sequences (Iterable[str]): Sequences.
    """
    with open(filepath, "w") as f:
        for header, sequence in zip(headers, sequences):
            f.write(f">{header}\n{sequence}\n")
