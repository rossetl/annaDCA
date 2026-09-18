import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from annadca.dataset import annaDataset


def run(*args):
    return subprocess.run([sys.executable, "-m", "annadca", *args], capture_output=True, text=True)


def test_cli():
    assert run("--help").returncode == 0
    assert run("--version").returncode == 0
    assert run("invalid").returncode != 0
    assert run("train").returncode != 0


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("marginalize", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
)])
def test_train_sample(tmp_path, binary, marginalize, device):
    data = tmp_path / "data.csv"
    pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "sequence": ["010", "110", "001", "101"] if binary else ["ACA", "CAC", "AAC", "CCA"],
            "label": ["x", "y", "" if marginalize else "x", "y"],
        }
    ).to_csv(data, index=False)
    flags = ["--is_binary"] if binary else ["--alphabet", "AC"]
    result = run(
        "train",
        "-d",
        str(data),
        "-o",
        str(tmp_path / "model"),
        "--device",
        device,
        *(["--no-triton"] if device == "cpu" else []),
        "--hidden",
        "2",
        "--nchains",
        "2",
        "--nepochs",
        "2",
        "--gibbs_steps",
        "1",
        "--no_reweighting",
        *(["--marginalize_missing_labels"] if marginalize else []),
        *flags,
    )
    assert result.returncode == 0, result.stderr
    model = tmp_path / "model" / "params.h5"
    with h5py.File(model) as f:
        assert "update_2" in f
        assert "label_vocabulary" in f.attrs
    # A subset of labels must retain its original model index.
    frame = pd.read_csv(data, dtype=str).iloc[[1, 3]]
    frame.to_csv(data, index=False)
    output = tmp_path / "samples.csv"
    result = run(
        "sample",
        "-d",
        str(data),
        "-p",
        str(model),
        "-o",
        str(output),
        "--device",
        "cpu",
        "-g",
        "1",
        *flags,
    )
    assert result.returncode == 0, result.stderr
    samples = pd.read_csv(output, dtype=str)
    assert samples.label.tolist() == ["y", "y"]
    assert samples.sequence.str.len().tolist() == [3, 3]


def test_binary_text_custom_columns(tmp_path):
    data = tmp_path / "data.txt"
    np.savetxt(data, [[0, 1, 0]])
    annotations = tmp_path / "labels.csv"
    annotations.write_text("ID,Class\n0,NA\n")
    dataset = annaDataset(
        data,
        str(annotations),
        column_names="ID",
        column_sequences="Seq",
        column_labels="Class",
        is_binary=True,
    )
    assert torch.equal(dataset.data_one_hot, torch.tensor([[0.0, 1.0, 0.0]]))
    assert dataset.idx_to_label == {0: "NA"}


def test_unaligned_sequences(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("name,sequence,label\na,AC,x\nb,A,y\n")
    with pytest.raises(ValueError, match="aligned"):
        annaDataset(data)


def test_training_triton_flags():
    from annadca.scripts.train import create_parser

    parser = create_parser()
    required = ["-d", "unused.csv", "-o", "unused"]
    assert parser.parse_args(required).triton
    assert parser.parse_args([*required, "--triton"]).triton
    assert not parser.parse_args([*required, "--no-triton"]).triton
    result = run("train", *required, "--device", "cpu")
    assert result.returncode != 0
    assert "--no-triton" in result.stderr
    assert "requires float32 CUDA" in result.stderr
