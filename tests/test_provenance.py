from ao_lab.evaluate import sha256_file


def test_sha256_changes_with_content(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    a = sha256_file(p)
    p.write_bytes(b"abd")
    b = sha256_file(p)
    assert a != b
