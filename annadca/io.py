from typing import Dict, Optional
import torch
from torch.nn.functional import one_hot
from adabmDCA.fasta import encode_sequence
from annadca.utils.fasta import read_fasta, write_fasta


def save_chains(
    filepath: str,
    chains: Dict[str, torch.Tensor],
    tokens: str,
    idx_to_label: Optional[dict] = None,
):
    """Saves the visible and label configurations of the chains to a fasta file.
    The header of each sequence has the form 'chain_<i>|label=<label>'.

    Args:
        filepath (str): Path to the output fasta file.
        chains (Dict[str, torch.Tensor]): Chains with keys 'visible' and 'label'. The visible configurations must be binary (N, L)
            or one-hot encoded (N, L, q). The labels must be one-hot (N, num_classes) or continuous (N, 1).
        tokens (str): Alphabet used to encode the sequences ('01' for binary variables).
        idx_to_label (dict, optional): Map from the label index to the label name. If None, the label index is saved.
            Ignored for continuous labels.
    """
    visible, label = chains["visible"], chains["label"]
    if visible.dim() == 3:
        visible = visible.argmax(dim=-1)
    visible = visible.long().cpu().numpy()
    if label.shape[-1] == 1 and not torch.all((label == 0) | (label == 1)):
        label_values = [f"{x:.6g}" for x in label.view(-1).tolist()]
    else:
        # all-zero vectors (missing labels) are saved as 'None'
        label_values = [str(idx_to_label[i] if idx_to_label is not None else i) if valid else "None"
                        for i, valid in zip(label.argmax(dim=-1).tolist(), (label.sum(dim=-1) > 0).tolist())]
    headers = [f"chain_{i}|label={l}" for i, l in enumerate(label_values)]
    sequences = ["".join(tokens[x] for x in seq) for seq in visible]
    write_fasta(filepath, headers, sequences)


def load_chains(
    filepath: str,
    tokens: str,
    num_classes: int,
    label_to_idx: Optional[dict] = None,
    continuous_labels: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Loads the chains saved with `save_chains`.

    Args:
        filepath (str): Path to the fasta file.
        tokens (str): Alphabet used to encode the sequences ('01' for binary variables).
        num_classes (int): Number of label classes.
        label_to_idx (dict, optional): Map from the label name (as a string) to the label index. If None, the headers are assumed to contain the label index.
        continuous_labels (bool, optional): Whether the labels are continuous. Defaults to False.
        device (torch.device, optional): Device of the output tensors. Defaults to cpu.
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float32.

    Returns:
        Dict[str, torch.Tensor]: Chains with keys 'visible' and 'label'.
    """
    headers, sequences = read_fasta(filepath)
    visible = torch.tensor(encode_sequence(sequences, tokens), device=device)
    if tokens == "01":
        visible = visible.to(dtype)
    else:
        visible = one_hot(visible.long(), num_classes=len(tokens)).to(dtype)

    label_values = [h.split("|label=")[-1] for h in headers]
    if continuous_labels:
        label = torch.tensor([float(l) for l in label_values], device=device, dtype=dtype).view(-1, 1)
    else:
        label = torch.zeros((len(headers), num_classes), device=device, dtype=dtype)
        if label_to_idx is not None:
            label_to_idx = {str(k): v for k, v in label_to_idx.items()}
        for n, l in enumerate(label_values):
            if l != "None":
                label[n, label_to_idx[l] if label_to_idx is not None else int(l)] = 1.0
    return {"visible": visible, "label": label}
