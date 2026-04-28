from typing import Dict, Tuple

import os
import h5py
import numpy as np
import torch
from adabmDCA.fasta import write_fasta, get_tokens
from rbms.io import load_model
from rbms.utils import get_saved_updates as get_saved_updates_ptt

from annadca.utils import get_saved_updates

def _save_model(
    params: Dict[str, torch.Tensor],
    filename: str,
    num_updates: int,
):
    """Save the current state of the model.

    Args:
        params (Dict[str, torch.Tensor]): Parameters of the model.
        filename (str): Path to the h5 archive where to store the model.
        num_updates (int): Number of updates performed so far.
    """
    # overwrite the file with a new archive containing only the current checkpoint
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with h5py.File(filename, "w") as f:
        checkpoint = f.create_group(f"update_{num_updates}")

        # Save the parameters of the model
        checkpoint["vbias"] = params["vbias"].detach().cpu().numpy()
        checkpoint["hbias"] = params["hbias"].detach().cpu().numpy()
        checkpoint["lbias"] = params["lbias"].detach().cpu().numpy()
        checkpoint["weight_matrix"] = params["weight_matrix"].detach().cpu().numpy()
        checkpoint["label_matrix"] = params["label_matrix"].detach().cpu().numpy()

        # Save current random state
        checkpoint["torch_rng_state"] = np.array(torch.get_rng_state(), dtype="uint8")
        np_state = np.random.get_state()
        checkpoint["numpy_rng_arg0"] = np_state[0]
        checkpoint["numpy_rng_arg1"] = np_state[1]
        checkpoint["numpy_rng_arg2"] = np_state[2]
        checkpoint["numpy_rng_arg3"] = np_state[3]
        checkpoint["numpy_rng_arg4"] = np_state[4]
    

def _load_model(
    filename: str,
    device: torch.device,
    dtype: torch.dtype,
    index: int | None = None,
    set_rng_state: bool = False,
) -> Tuple[int, Dict[str, torch.Tensor]]:
    """Loads a RBM from an h5 archive.

    Args:
        filename (str): Path to the h5 archive.
        device (torch.device): PyTorch device on which to load the parameters and the chains.
        dtype (torch.dtype): Dtype for the parameters and the chains.
        index (int | None, optional): Index of the machine to load. If None, the last machine is loaded. Defaults to None.
        set_rng_state (bool, optional): Restore the random state at the given epoch (useful to restore training). Defaults to False.

    Returns:
        Tuple[int, Dict[str, torch.Tensor]]: Number of updates and parameters of the loaded model.
    """
    list_updates = get_saved_updates(filename)
    if index is None:
        index = list_updates[-1]
    else:
        if index not in list_updates:
            raise ValueError(f"Index {index} not found in the h5 archive.")
    
    last_file_key = f"update_{index}"
    with h5py.File(filename, "r") as f:
        weight_matrix = torch.tensor(
            f[last_file_key]["weight_matrix"][()],
            device=device,
            dtype=dtype,
        )
        label_matrix = torch.tensor(
            f[last_file_key]["label_matrix"][()],
            device=device,
            dtype=dtype,
        )
        vbias = torch.tensor(f[last_file_key]["vbias"][()], device=device, dtype=dtype)
        hbias = torch.tensor(f[last_file_key]["hbias"][()], device=device, dtype=dtype)
        lbias = torch.tensor(f[last_file_key]["lbias"][()], device=device, dtype=dtype)
        
        if set_rng_state:
            torch.set_rng_state(torch.tensor(np.array(f[last_file_key]["torch_rng_state"])))
            np_rng_state = tuple(
                [
                    f[last_file_key]["numpy_rng_arg0"][()].decode("utf-8"),
                    f[last_file_key]["numpy_rng_arg1"][()],
                    f[last_file_key]["numpy_rng_arg2"][()],
                    f[last_file_key]["numpy_rng_arg3"][()],
                    f[last_file_key]["numpy_rng_arg4"][()],
                ]
            )
            np.random.set_state(np_rng_state)
        
    params = {
        "weight_matrix": weight_matrix,
        "label_matrix": label_matrix,
        "vbias": vbias,
        "hbias": hbias,
        "lbias": lbias,
    }
    
    return index, params

def _load_model_from_ptt(
    filename: str,
    index: int | None,
    device: torch.device,
    dtype: torch.dtype,
    num_labels: int,
    label_frequencies: torch.Tensor | None = None,
) -> Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Loads a RBM from a ptt file.

    Args:
        filename (str): Path to the ptt file.
        index (int | None): Index of the machine to load. If None, the last machine is loaded.
        device (torch.device): PyTorch device on which to load the parameters and the chains.
        dtype (torch.dtype): Dtype for the parameters and the chains.
        num_labels (int): Number of label classes for the annaRBM.
        label_frequencies (torch.Tensor | None): Label frequencies for initialization. Defaults to None.

    Returns:
        Tuple[int, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Update index, parameters and chains of the loaded model.
    """
    if index is None:
        index: int = get_saved_updates_ptt(filename)[-1]
    params_ptt, perm_chains_ptt, _ = load_model(filename, index, device, dtype)
    vbias = params_ptt["vbias"] if isinstance(params_ptt, dict) else params_ptt.vbias
    hbias = params_ptt["hbias"] if isinstance(params_ptt, dict) else params_ptt.hbias
    weight_matrix = params_ptt["weight_matrix"] if isinstance(params_ptt, dict) else params_ptt.weight_matrix
    is_binary = weight_matrix.dim() == 2
    H = weight_matrix.shape[-1]
    params = {
        "vbias": vbias,
        "hbias": hbias,
        "weight_matrix": weight_matrix,
    }
    if label_frequencies is None:
        lbias = torch.zeros(num_labels, device=device, dtype=dtype)
    else:
        label_frequencies = label_frequencies.to(device=device, dtype=dtype)
        label_frequencies = torch.clamp(label_frequencies, min=1e-8, max=1.0 - 1e-8)
        lbias = torch.log(label_frequencies) - torch.log(1.0 - label_frequencies)
    params["lbias"] = lbias
    params["label_matrix"] = torch.randn(num_labels, H, device=device, dtype=dtype) * 1e-4

    if is_binary:
        perm_chains_ptt["visible"] = perm_chains_ptt["visible"].to(device=device, dtype=dtype)
    else:
        q = weight_matrix.shape[1]
        chains_visible_onehot = torch.nn.functional.one_hot(
            perm_chains_ptt["visible"].long(),
            num_classes=q,
        ).to(device=device, dtype=dtype)
        perm_chains_ptt["visible"] = chains_visible_onehot

    for key in ["hidden", "label"]:
        if key in perm_chains_ptt:
            perm_chains_ptt[key] = perm_chains_ptt[key].to(device=device, dtype=dtype)

    return index, params, perm_chains_ptt
    
    
def _save_chains(
    filename: str,
    visible: torch.Tensor,
    label: torch.Tensor,
    alphabet: str,
) -> None:
    """Save the persistent chains on a fasta file.
    
    Args:
        filename (str): Path to the fasta file.
        visible (torch.Tensor): Visible units of the chains.
        label (torch.Tensor): Labels of the chains.
        alphabet (str): Alphabet to be used for the encosing of the sequences.
    """
    tokens = get_tokens(alphabet)
    visible = visible.int().cpu().numpy()
    label = label.int().cpu().numpy()
    # Headers are associated with the labels
    headers = np.vectorize(lambda x: "".join([str(i) for i in x]), signature="(l) -> ()")(label)
    write_fasta(
        fname=filename,
        headers=headers,
        sequences=visible,
        tokens=tokens,
        remove_gaps=False,
    )

