"""
Lever #2 — structured metadata filtering + aggregation for the Diavgeia assistant.

Turns countable questions ("how many...", "count by...") into exact Elasticsearch
aggregations over the indexed metadata, with IDs resolved to human names via the
public Diavgeia OpenData API. Added in the 2026-07 upgrade session; kept in its own
package so it sits cleanly alongside the existing retrieval/generation code.
"""
