import numpy as np
import pandas as pd
import torch
from annadca.dataset import annaDataset
from annadca.utils.fasta import write_fasta


def test_csv_categorical(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "header": ["s1", "s2", "s3", "s4"],
        "sequence": ["AC-D", "ACCD", "AXCD", "GC-D"],  # 'X' is not in the alphabet
        "label": ["a", "b", "a", None],
    }).to_csv(path, index=False)
    dataset = annaDataset(str(path), column_names="header", alphabet="protein")
    assert len(dataset) == 3
    assert dataset.data_one_hot.shape == (3, 4, 21)
    assert dataset.get_num_classes() == 2
    # Missing labels are encoded as all-zero vectors
    assert torch.equal(dataset.labels_one_hot.sum(1), torch.tensor([1.0, 1.0, 0.0]))
    assert list(dataset.to_label(dataset.labels_one_hot[:2])) == ["a", "b"]


def test_fasta_with_annotations(tmp_path):
    fasta, ann = tmp_path / "data.fasta", tmp_path / "ann.csv"
    write_fasta(str(fasta), ["s1", "s2", "s3"], ["ACGU", "ACGG", "UCGU"])
    pd.DataFrame({"id": ["s3", "s1", "s2"], "y": [1, 0, 1]}).to_csv(ann, index=False)
    dataset = annaDataset(str(fasta), path_ann=str(ann), column_names="id", column_labels="y", alphabet="rna")
    assert dataset.data_one_hot.shape == (3, 4, 5)
    assert list(dataset.to_label(dataset.labels_one_hot)) == [0, 1, 1]


def test_text_binary_with_annotations(tmp_path):
    data, ann = tmp_path / "data.txt", tmp_path / "ann.csv"
    np.savetxt(data, np.array([[0, 1, 1], [1, 0, 0], [1, 1, 1]]), fmt="%d")
    pd.DataFrame({"name": ["0", "1", "2"], "label": ["x", "y", "x"]}).to_csv(ann, index=False)
    dataset = annaDataset(str(data), path_ann=str(ann), is_binary=True)
    assert torch.equal(dataset.data_one_hot, torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    assert list(dataset.to_label(dataset.labels_one_hot)) == ["x", "y", "x"]


def test_csv_binary_keeps_leading_zeros(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"name": ["a", "b"], "sequence": ["0011", "0101"], "label": [0, 1]}).to_csv(path, index=False)
    dataset = annaDataset(str(path), is_binary=True)
    assert torch.equal(dataset.data_one_hot, torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]))


def test_continuous_labels(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "name": ["a", "b", "c", "d"],
        "sequence": ["ACDE", "ACDF", "ACDG", "ACDH"],
        "label": [1.0, 2.0, 3.0, None],
    }).to_csv(path, index=False)
    dataset = annaDataset(str(path), continuous_labels=True)
    assert dataset.get_num_classes() == 1
    assert torch.isfinite(dataset.labels_one_hot).all()
    assert np.allclose(dataset.to_label(dataset.labels_one_hot[:3]), [1.0, 2.0, 3.0])
