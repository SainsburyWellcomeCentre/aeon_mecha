# Spike-Sorting Curation: Two Design Decisions for the Team

**Prepared:** 2026-08-26 (for the 2026-08-27 meeting)
**Context:** Elissa's planned PR **into Zofia's `zs/spike_sorting` branch** (PR #604), landing before
#604 merges to `main`. Both issues below were found while reviewing #604 with Zofia.
**Why bring it to the team:** both decisions are core/shared design (Thinh) and touch Zofia's
in-progress curation work, so they're worth deciding together rather than patching unilaterally.

> **How to read this**
> - **[Verified]** — checked against the current branch code, GitHub `main`, and the original spec.
> - **[Proposal]** — Claude's analysis of options and trade-offs, for discussion. Not decisions.
> - **[Decided — Elissa]** — calls Elissa has already made.

---

## Background — two problems in the curation pipeline

### Problem A — the manual quality label is never stored [Verified]

- `SortedSpikes.Unit.unit_quality` is meant to carry each unit's quality label.
- In every code path that builds those rows — `SortedSpikes.make()`, and on the branch also
  `insert_sorted_spikes_from_analyzer()` — `unit_quality` is read from Kilosort's `KSLabel`
  property, never from the curator's manual `quality` property. So it always reflects Kilosort's
  automatic call, never the human's curation, even after a curation has been applied.
- The original spec (`docs/specs/SPEC_SPIKE_SORTING_CURATION.md`, unchanged since 2026-04-20)
  describes a **single** quality field that holds Kilosort's label before curation and is
  **replaced** by the curator's label once a curation is applied. There is no second field in it.
- This is a **pre-existing bug in `main`**, not something Zofia introduced.
- Mechanism (verified): SpikeInterface stores the curator's call in a separate `quality` property
  and leaves `KSLabel` untouched, so any code reading `KSLabel` never picks up the curation. Per
  Elissa, the original code read `KSLabel` on the assumption that applying a curation overwrites it
  in place — the separate `quality` property wasn't understood at the time.
- On the branch, Zofia added a **second** column, `curation_quality`, to hold the manual label
  separately, plus a `UnitTag` part table for non-exclusive tags. The spec was never updated to
  describe either.

### Problem B — applying a curation deletes its own parent (circular cascade) [Verified]

- Dependency chain: `PostProcessing → SortedSpikes → OfficialCuration → ApplyOfficialCuration`
  (each table borrows its primary key from the one above).
- `ApplyOfficialCuration.make()` deletes the `SortedSpikes` row in order to swap raw units for
  curated ones. That delete cascades **down the chain**: it removes the `OfficialCuration` child,
  which in turn removes the `ApplyOfficialCuration` grandchild.
- The next line then tries to insert the `ApplyOfficialCuration` record — but the `OfficialCuration`
  parent it must attach to was just cascade-deleted. Foreign-key violation. The step cannot complete.
- This is **fatal and pre-existing in `main`**. It was never caught because the apply step was a
  draft that was never actually run until Zofia tested it.
- The spec's "Circular Dependency Handling" section addresses only the Python **import** cycle
  (the lazy import), not this table cascade.
- **Zofia's workaround:** inside `make()`, stash the `OfficialCuration` row in memory before the
  delete, rebuild `SortedSpikes` from the in-memory analyzer, then re-insert `OfficialCuration` and
  `ApplyOfficialCuration` in the same transaction. It works, but a computed table is reaching
  upstream to delete and rebuild its own ancestor.

---

## Decision 1 — how to represent quality labels [Proposal]

Fixing Problem A forces a choice: when `unit_quality` reflects the manual curation, do we keep
Kilosort's label too?

### Option 1A — single field (the original spec)
One `unit_quality` field: holds the curator's manual label once curated, with Kilosort's label
standing in until then.
- **Pros:** matches the spec; one source of truth; simplest; removes Zofia's extra column, moving
  the branch back toward `main`.
- **Cons:** Kilosort's label is gone once a unit is curated (no side-by-side comparison); a bare
  label doesn't say whether it's a human or a machine call unless we mark it.
- **Sub-choice:** mark Kilosort-sourced values with a `ks_` prefix (`ks_good` / `ks_mua` /
  `ks_noise`) so provenance stays visible, or leave them bare.

### Option 1B — two fields (the branch's current state)
`unit_quality` = Kilosort's label always; `curation_quality` = the manual label (null until curated).
- **Pros:** both labels visible at once; no ambiguity; nothing lost.
- **Cons:** diverges from the spec; two fields to keep straight everywhere downstream; "fixes" the
  bug by adding a field rather than correcting the one that was wrong.

> Related, **[Decided — Elissa]**: noise-labelled units are **kept and labelled**, not deleted, and
> are **excluded from unit matching**. That holds under either option above; only which field the
> matching filter reads would change.

---

## Decision 2 — in-place modification vs. duplication [Proposal]

Fixing Problem B forces a choice about how curated results are stored. Note (Elissa's framing): every
option short of 2B still feels like a patch, because the underlying issue is that one mutable table
holds both the raw and the curated result and swaps between them in place.

### Option 2A — keep in-place modification, patch the cascade
One `SortedSpikes` table, swapped between raw and curated; fix only the circularity.
- **Variant i — reparent:** move `OfficialCuration`'s parent from `SortedSpikes` up to
  `PostProcessing` (same primary key, since they're 1:1). Deleting `SortedSpikes` no longer destroys
  the curation records, and Zofia's in-memory recreate can be removed.
- **Variant ii — keep Zofia's in-memory recreate** as the workaround.
- **Pros:** minimal; downstream tables untouched; unblocks Zofia now.
- **Cons:** treats the symptom, not the cause — a computed table still deletes and rebuilds an
  upstream table in place. The reparent also loosens the enforced link between "which curation is
  official" and "what's actually in `SortedSpikes` right now" (left to the `curation_id` convention).
  Still a schema change on the live DB.

### Option 2B — duplication (DataJoint Elements pattern)
Raw `SortedSpikes` becomes immutable. The curated result lives in its own downstream table computed
from `OfficialCuration`; downstream reads the curated table, falling back to raw when there's no
curation. Nothing is ever deleted and rebuilt.
- **Pros:** removes both the circularity and the in-place mutation entirely; matches the established
  DataJoint Elements curation pattern (`Clustering → Curation → CuratedClustering`); clean lineage.
- **Cons:** larger restructure; repoints downstream tables (`SyncedSpikes`, `UnitMatching`, …); a
  second units table with some duplicated extraction logic; migration on the live DB.

---

## Suggested framing for the discussion [Proposal]

- Both problems are pre-existing in `main`, so fixing them here also improves what eventually lands
  in `main`.
- The two decisions interact: the cleaner Decision-2 design (2B) is also the natural moment to settle
  Decision 1, since the curated units table would be built fresh either way.
- **Sensitivity:** Zofia is already running the branch on real curated data. A restructure (2B) or
  the schema changes (1A / 2A) may require her to re-apply or re-curate existing blocks — worth
  checking what she's willing to redo before choosing.

## Open questions for the team
1. **Decision 1:** single field (1A — with or without the `ks_` prefix) or two fields (1B)?
2. **Decision 2:** patch in place now (2A-i, reparent) or restructure to duplication (2B)?
3. If 2B: do it now, or patch with 2A to unblock and schedule 2B as the planned "full redesign"?
4. How much of her existing curated data is Zofia willing to re-apply / re-curate afterward?
