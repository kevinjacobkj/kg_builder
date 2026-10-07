import os
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from text_files.text_parser import Chunk, load_chunks  # noqa: E402

DEFAULT_MODEL = "gpt-5.4-mini"

INSTRUCTIONS = """You extract named entities from text for a knowledge graph.
Return every distinct named entity mentioned in the text, each exactly once, using its most complete surface form.
Entity types:
- PER: specific people
- ORG: companies, institutions, and named products or platforms owned by an organization
- LOC: cities, states, countries, and other geographic places
- MISC: other proper nouns that do not fit the above
Do not include generic nouns, job titles, dates, or department names unless they are proper names."""


class Entity(BaseModel):
    type: Literal["PER", "ORG", "LOC", "MISC"]
    name: str


class EntityList(BaseModel):
    entities: list[Entity]


def extract_chunk_entities(client: OpenAI, chunk: Chunk, model: str) -> list[Entity]:
    response = client.responses.parse(
        model=model,
        instructions=INSTRUCTIONS,
        input=chunk.text,
        text_format=EntityList,
    )
    return response.output_parsed.entities


def extract_entities(chunks: list[Chunk], client: OpenAI, model: str) -> dict[str, list[Entity]]:
    return {chunk.id: extract_chunk_entities(client, chunk, model) for chunk in chunks}


def write_entities(entities: dict[str, list[Entity]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for chunk_id, results in entities.items():
        lines.append(f"[{chunk_id}]")
        for ent in results:
            lines.append(f"{ent.type}\t{ent.name}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")

    input_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "chunks.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent / "llm-entities.txt"
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    chunks = load_chunks(input_path)
    entities = extract_entities(chunks, OpenAI(), model)
    write_entities(entities, output_path)
    total = sum(len(v) for v in entities.values())
    print(f"Wrote {total} entities from {len(chunks)} chunks to {output_path} using {model}")
