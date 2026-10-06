import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

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


def write_chunks(chunks: list[Chunk], path: str | Path = "chunks.json") -> None:
    data = [asdict(chunk) for chunk in chunks]
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: str | Path = "chunks.json") -> list[Chunk]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk(**item) for item in data]


if __name__ == "__main__":
    input_path = Path(sys.argv[1] if len(sys.argv) > 1 else "text_files")
    output_path = sys.argv[2] if len(sys.argv) > 2 else "chunks.json"

    chunks = parse_directory(input_path) if input_path.is_dir() else parse_file(input_path)
    write_chunks(chunks, output_path)
    print(f"Wrote {len(chunks)} chunks from {input_path} to {output_path}")
