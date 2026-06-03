"""Minimal example that does not require Crafter."""

from embryo.memory.facts import facing_candidate_fact
from embryo.memory.queries import query_content_hash


fact = facing_candidate_fact(True, confidence=1.0)
print(query_content_hash([fact]))
