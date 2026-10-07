import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import datasets
from fastcoref.coref_models.modeling_fcoref import FCorefModel
from fastcoref.modeling import CorefModel
from fastcoref.utilities.collate import LeftOversCollator

COREF_MODEL = "biu-nlp/f-coref"
SPACY_MODEL = "en_core_web_sm"
# Fraction of a coref mention's characters an NER entity must cover to be treated as that mention.
MIN_COVERAGE = 0.5

Span = tuple[int, int]

# fastcoref configures root logging at INFO on import, which floods the console with HTTP request logs.
logging.getLogger().setLevel(logging.WARNING)
datasets.disable_progress_bars()
warnings.filterwarnings("ignore", category=FutureWarning, module="pyarrow")


class _FCorefModel(FCorefModel):
    """fastcoref calls the pre-5.x `init_weights()`; transformers 5 also needs `post_init()` to load weights."""

    def __init__(self, config):
        super().__init__(config)
        self.post_init()


@dataclass
class Mention:
    type: str
    name: str
    score: float
    start: int
    end: int


@dataclass
class Entity:
    type: str
    name: str
    score: float
    ner_spans: list[Span] = field(default_factory=list)
    coref_spans: list[Span] = field(default_factory=list)

    @property
    def spans(self) -> list[Span]:
        return sorted(self.ner_spans + self.coref_spans)


def build_coref_model(model_name: str = COREF_MODEL, device: str = "cpu") -> CorefModel:
    return CorefModel(model_name, _FCorefModel, LeftOversCollator, False, device, SPACY_MODEL)


def predict_clusters(texts: list[str], model: CorefModel) -> list[list[list[Span]]]:
    return [result.get_clusters(as_strings=False) for result in model.predict(texts=texts)]


def _tokens(name: str) -> list[str]:
    return name.lower().removesuffix("'s").rstrip("'").split()


def _compatible(a: str, b: str) -> bool:
    """True if one name's tokens are contained in the other's (e.g. 'David' and 'David Wilson')."""
    short, long = sorted((_tokens(a), _tokens(b)), key=len)
    return set(short) <= set(long)


def _overlaps(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _coverage(mention: Mention, span: Span) -> float:
    overlap = min(mention.end, span[1]) - max(mention.start, span[0])
    return max(overlap, 0) / (span[1] - span[0])


def _is_head(mention: Mention, span: Span, text: str) -> bool:
    """True for appositive spans like 'Orion Cloud, a platform ...' that open with the entity."""
    if mention.start < span[0] or mention.end > span[1]:
        return False
    prefix = text[span[0]:mention.start].lower()
    return prefix in ("", "the ") and (mention.end == span[1] or text[mention.end] == ",")


def _match_span(span: Span, mentions: list[Mention], text: str) -> Mention | None:
    candidates = [m for m in mentions if _overlaps((m.start, m.end), span)]
    best = max(candidates, key=lambda m: _coverage(m, span), default=None)
    if best and _coverage(best, span) >= MIN_COVERAGE:
        return best
    return next((m for m in candidates if _is_head(m, span, text)), None)


def _group_compatible(mentions: list[Mention]) -> list[list[Mention]]:
    """Split one cluster's NER mentions into groups whose names refer to the same entity."""
    groups: list[list[Mention]] = []
    for mention in sorted(mentions, key=lambda m: -len(_tokens(m.name))):
        group = next((g for g in groups if _compatible(g[0].name, mention.name)), None)
        if group is None:
            groups.append([mention])
        else:
            group.append(mention)
    return groups


def _canonical(group: list[Mention]) -> Entity:
    names = Counter(m.name for m in group)
    name = max(names, key=lambda n: (len(_tokens(n)), names[n], len(n)))
    type_scores: dict[str, float] = defaultdict(float)
    for m in group:
        type_scores[m.type] += m.score
    entity_type = max(type_scores, key=type_scores.get)
    return Entity(entity_type, name, max(m.score for m in group), [(m.start, m.end) for m in group])


def _resolve_cluster(cluster: list[Span], mentions: list[Mention], text: str) -> list[Entity]:
    """Attribute each span to the entity of the nearest preceding NER-matched span (or the first, for cataphora),
    so a cluster that wrongly links two differently named entities is split rather than merged."""
    matched = [(span, _match_span(span, mentions, text)) for span in sorted(cluster)]
    ner_mentions = list({id(m): m for _, m in matched if m}.values())
    if not ner_mentions:
        return []

    groups = _group_compatible(ner_mentions)
    entities = [_canonical(g) for g in groups]
    group_of = {id(m): i for i, g in enumerate(groups) for m in g}

    current = group_of[id(next(m for _, m in matched if m))]
    for span, mention in matched:
        if mention:
            current = group_of[id(mention)]
        else:
            entities[current].coref_spans.append(span)
    return entities


def _merge_same_name(entities: list[Entity]) -> list[Entity]:
    merged: dict[tuple[str, str], Entity] = {}
    for entity in entities:
        key = (entity.type, entity.name)
        if key not in merged:
            merged[key] = Entity(entity.type, entity.name, entity.score)
        target = merged[key]
        target.score = max(target.score, entity.score)
        target.ner_spans.extend(s for s in entity.ner_spans if s not in target.ner_spans)
        target.coref_spans.extend(s for s in entity.coref_spans if s not in target.coref_spans)

    for entity in merged.values():
        entity.coref_spans = [
            s for s in entity.coref_spans if not any(_overlaps(s, n) for n in entity.ner_spans)
        ]
    return list(merged.values())


def resolve_entities(text: str, mentions: list[Mention], clusters: list[list[Span]]) -> list[Entity]:
    """Merge NER mentions linked by coref clusters into entities, attaching pronoun/nominal mentions too."""
    entities = [e for cluster in clusters for e in _resolve_cluster(cluster, mentions, text)]
    clustered = {s for e in entities for s in e.ner_spans}
    entities += [
        Entity(m.type, m.name, m.score, [(m.start, m.end)])
        for m in mentions
        if (m.start, m.end) not in clustered
    ]
    return _merge_same_name(entities)
