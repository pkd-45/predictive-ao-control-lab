from ao_lab.config import load_config
from ao_lab.dataset import generate_dataset, load_split


def test_splits_have_disjoint_seeds(tmp_path):
    cfg = load_config("configs/baseline.yaml")
    path = generate_dataset(cfg, tmp_path / "d.h5")
    train = set(load_split(path, "train")[1].tolist())
    val = set(load_split(path, "val")[1].tolist())
    test = set(load_split(path, "test")[1].tolist())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
