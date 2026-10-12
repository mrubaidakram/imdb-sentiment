from pathlib import Path, PurePosixPath
import tarfile
import urllib.request

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE = "aclImdb_v1.tar.gz"

def _safe_extract(archive: Path, destination: Path):
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not name.parts or name.parts[0] != "aclImdb":
                raise ValueError(f"Unsafe archive member: {member.name}")
            if not (member.isfile() or member.isdir()):
                continue
            target = (root / Path(*name.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive path escapes destination: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                source = tar.extractfile(member)
                if source is None: raise ValueError(f"Unreadable archive member: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output: output.write(source.read())

def ensure_data(root: str):
    root = Path(root); corpus = root / "aclImdb"
    if corpus.is_dir(): return corpus
    root.mkdir(parents=True, exist_ok=True); archive = root / ARCHIVE
    if not archive.exists():
        temporary = root / (ARCHIVE + ".partial")
        try:
            urllib.request.urlretrieve(URL, temporary)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    _safe_extract(archive, root)
    if not corpus.is_dir(): raise RuntimeError("The downloaded archive did not contain aclImdb/")
    return corpus

def load_split(root: Path, split: str):
    texts, labels = [], []
    for label, name in ((0, "neg"), (1, "pos")):
        folder = root / split / name
        if not folder.is_dir(): raise FileNotFoundError(f"Missing labeled review folder: {folder}")
        for path in sorted(folder.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8", errors="replace")); labels.append(label)
    return texts, labels
