from typing import Dict

import h5py
import numpy as np
import torch
from adabmDCA.fasta_utils import write_fasta, get_tokens

from aiDCA.utils import get_saved_updates

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

    with h5py.File(filename, "a") as f:
        checkpoint = f.create_group(f"update_{num_updates}")

        # Save the parameters of the model
        checkpoint["vbias"] = params["vbias"].detach().cpu().numpy()
        checkpoint["hbias"] = params["hbias"].detach().cpu().numpy()
        checkpoint["lbias"] = params["lbias"].detach().cpu().numpy()
        checkpoint["weight_matrix"] = params["weight_matrix"].detach().cpu().numpy()
        checkpoint["label_matrix"] = params["label_matrix"].detach().cpu().numpy()
        

        # Save current random state
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
        checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
        checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
        checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
        checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]


def _load_model(
    filename: str,
    device: torch.device,
    dtype: torch.dtype,
    index: int = None,
    set_rng_state: bool = False,
) -> Dict[str, torch.Tensor]:
    """Loads a RBM from an h5 archive.

    Args:
        filename (str): Path to the h5 archive.
        device (torch.device): PyTorch device on which to load the parameters and the chains.
        dtype (torch.dtype): Dtype for the parameters and the chains.
        index (int): Index of the machine to load. If None, the last machine is loaded. Defaults to None.
        set_rng_state (bool): Restore the random state at the given epoch (useful to restore training). Defaults to False.

    Returns:
        Dict[str, torch.Tensor]: Parameters of the loaded model.
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
    
    return params


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
        numeric_input=True,
        alphabet=tokens,
        remove_gaps=False,
    )

