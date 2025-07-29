from pathlib import Path
import hashlib


def blake2b_file(path, buf_size=128_000, digest_size=32):
    """Calculate the BLAKE2b hash of a file.

    Args:
        path (str): Path to the file to hash.
        buf_size (int): Buffer size for reading the file in chunks.
        digest_size (int): Size of the hash digest in bytes.
    Returns:
        str: The BLAKE2b hash of the file as a hexadecimal string.
    """
    h = hashlib.blake2b(digest_size=digest_size)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()
