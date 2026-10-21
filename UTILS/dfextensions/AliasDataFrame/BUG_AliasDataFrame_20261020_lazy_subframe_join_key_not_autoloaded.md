# BUG_AliasDataFrame_20261020 — Lazy recursive subframe join key is not auto-loaded

**Date:** 2026-10-20  
**Reporter:** GPT51:ADF / PHASE_13_76 commissioning  
**Owner:** AliasDataFrame  
**Status:** OPEN / PINNED KNOWN GAP

## Summary

A lazy ROOT-backed AliasDataFrame can register a keyed subframe and accept a
qualified recursive alias while the root join key is still unloaded. Final
recursive materialization then raises `KeyError` because the join key has not
been loaded automatically.

When the join key is explicitly loaded first, the same recursive graph
materializes exactly and the remaining value branch is loaded on demand.

## Current contract pinned by PHASE_13_76 v0.4

```text
lazy root, join key unloaded:
    recursive materialization -> KeyError

fresh lazy root:
    ensure_columns("oracle_row_id")
    then recursive public draw -> exact truth
```

PHASE_13_76 must not silently claim automatic lazy join-key discovery.

## Scope decision

Fixing automatic lazy join-key discovery is a production behavior change. The
bounded PHASE_13_76 continuation records and falsifies the current precondition
instead of expanding the cleanup phase with another production implementation.

## Future acceptance requirement

A later fix should allow the recursive request to succeed without explicit
join-key preload while preserving exact alignment, missing-key fill semantics,
lazy value-branch loading, A->B->C truth, and EAGER/LAZY parity.
