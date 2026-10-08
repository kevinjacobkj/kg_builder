import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = Path(__file__).resolve().parent / "chunks.json"
HF_DATASET = "usamaahmeddsn/dsn-boilerplate-all"
_SOURCE_FIELDS = ("record_id", "chunk_id", "block_id")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


def chunk_paragraphs(text: str, source: str) -> list[Chunk]:
    """Split text into one chunk per paragraph, with ids of the form '<source>_<n>'."""
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text)]
    return [Chunk(id=f"{source}_{i}", text=p) for i, p in enumerate(p for p in paragraphs if p)]


def parse_file(path: str | Path) -> list[Chunk]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return chunk_paragraphs(text, source=path.stem)


def parse_directory(directory: str | Path = "text_files", pattern: str = "*.txt") -> list[Chunk]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")

    chunks: list[Chunk] = []
    for path in sorted(directory.glob(pattern)):
        if path.is_file():
            chunks.extend(parse_file(path))
    return chunks


def parse_records(records: list[dict], stem: str) -> list[Chunk]:
    """Keep records that already have an 'id'; paragraph-chunk the rest under their record/block id."""
    chunks: list[Chunk] = []
    for n, record in enumerate(records):
        text = record.get("text") or ""
        if record.get("id"):
            chunks.append(Chunk(id=str(record["id"]), text=text))
        else:
            source = next((str(record[f]) for f in _SOURCE_FIELDS if record.get(f)), f"hf:{stem}#c{n}")
            chunks.extend(chunk_paragraphs(text, source))
    return chunks


def parse_hf_dataset(repo_id: str | None = None, token: str | None = None) -> list[Chunk]:
    """Download the dataset's .jsonl/.json files from Hugging Face and convert their records to chunks."""
    from dotenv import load_dotenv
    from huggingface_hub import HfApi, hf_hub_download

    load_dotenv(ROOT / ".env")
    repo_id = repo_id or os.environ.get("DSN_HF_DATASET") or HF_DATASET
    token = token or os.environ.get("DSN_HF_TOKEN")
    if not token:
        raise RuntimeError("DSN_HF_TOKEN is not set")

    files = sorted(f for f in HfApi(token=token).list_repo_files(repo_id, repo_type="dataset") if f.endswith((".jsonl", ".json")))
    if not files:
        raise FileNotFoundError(f"No .jsonl/.json files in dataset {repo_id}")

    chunks: list[Chunk] = []
    for name in files:
        text = Path(hf_hub_download(repo_id, name, repo_type="dataset", token=token)).read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()] if name.endswith(".jsonl") else json.loads(text)
        chunks.extend(parse_records(records, stem=Path(name).stem))
    return dedupe_chunks(chunks)


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact repeats; raise if one id maps to different texts."""
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.id in seen and seen[chunk.id] != chunk:
            raise ValueError(f"Duplicate chunk id with different text: {chunk.id}")
        seen.setdefault(chunk.id, chunk)
    return list(seen.values())


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_chunks(chunks: list[Chunk], path: str | Path = CHUNKS_PATH) -> None:
    data = [{"hash": chunk_hash(chunk.text), **asdict(chunk)} for chunk in chunks]
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: str | Path = CHUNKS_PATH) -> list[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(id=item["id"], text=item["text"]) for item in data]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        chunks = parse_hf_dataset()
        write_chunks(chunks, CHUNKS_PATH)
        print(f"Wrote {len(chunks)} chunks from Hugging Face to {CHUNKS_PATH}")
        sys.exit()

    input_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else CHUNKS_PATH

    chunks = parse_directory(input_path) if input_path.is_dir() else parse_file(input_path)
    write_chunks(chunks, output_path)
    print(f"Wrote {len(chunks)} chunks from {input_path} to {output_path}")
