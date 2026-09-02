# Spike-Sorting Curation — Meeting Reference (2026-08-27)

**Purpose:** Talking-points reference for the 2026-08-27 meeting on the spike-sorting curation
pipeline — Zofia's PR **#604** (`zs/spike_sorting`) and the follow-up PR **#608**
(`es/curation-fixes` → `zs/spike_sorting`).

> **Labels:** **[Verified]** — checked against the code, GitHub `main`, and the spec.
> **[Proposal]** — options for discussion, not decisions. **[Decided — Elissa]** — already decided,
> to confirm with the team.

---

## TL;DR — glance during the meeting

1. **Status** — 5 safe fixes shipped in PR #608 into `zs/spike_sorting`; the 2 big decisions below
   were deferred to today.
2. **Decision 1 — quality labels** — Bug: `unit_quality` is read from Kilosort's KSLabel and never
   reflects the manual curation. **Decide:** one field (Kilosort, replaced by the manual call) or two
   fields (keep both). The spec says one.
3. **Decision 2 — the cascade (fatal)** — Applying a curation deletes `SortedSpikes`, which
   cascade-deletes its own `OfficialCuration`/`ApplyOfficialCuration`, so the apply can't finish.
   Pre-existing in main, never actually worked. **Decide:** patch (reparent — quick) or restructure to
   a separate curated table (clean — bigger).
4. **Noise units** — Keep + label "noise" + exclude from matching (not delete). **Confirm** with
   Zofia/Thinh.
5. **`unit_quality` sub-choices** (if single field) — Use Kilosort as the per-unit fallback for
   unlabelled units? Mark machine labels with a `ks_` prefix?
6. **Label-completeness gate** — None exists today. **Decide:** add one, or rely on the fallback.
7. **Applying the changes = just repopulating** — No re-sorting and no re-curation: every sorting
   analyzer and curation file is saved, so it's only re-populating the tables, not real work for
   Zofia. **Be clear about that**, and settle with Zofia how she wants to handle re-applying.

*(Full detail on each is below if you need to drill in.)*

---

## 1. Status — what already landed in PR #608 (partial set) [Verified]

The safe, agreed fixes, so they don't block on the big design questions. All into `zs/spike_sorting`
before it reaches `main`:

- Removed the obsolete uncaught-exception workarounds (DataJoint 2.3.2 fixes it upstream).
- Fixed "2 tuples found" — name the file in the remaining `ManualCuration.File` reads.
- Made `save_manual_curation`'s two inserts atomic.
- Replaced `UnitMatching.CandidateMatch` with `UnitMatching.BlockComparison` (the full
  agreement-score grid per compared block, no threshold).
- Parameterized the curation tags into a `CurationTag` lookup table.

---

## 2. Decision 1 — how to represent quality labels

**The bug [Verified].** `SortedSpikes.Unit.unit_quality` is read from Kilosort's `KSLabel` in every
build path, never from the curator's manual `quality` property — so it always reflects Kilosort, never
the human's curation, even after a curation is applied. Pre-existing in `main`. The original spec
intended a **single** quality field: Kilosort's label before curation, replaced by the curator's after.
Zofia's branch instead added a **second** column, `curation_quality`, for the manual label.

**Option 1A — single field (the spec).** One `unit_quality`: the manual label once curated, Kilosort
standing in until then.
- Pros: matches the spec; one source of truth; simplest.
- Cons: Kilosort's label is lost once curated; a bare label is ambiguous (human vs. machine) unless
  marked with a `ks_` prefix.

**Option 1B — two fields (current branch).** `unit_quality` = Kilosort always; `curation_quality` =
manual (null until curated).
- Pros: both labels visible; nothing lost.
- Cons: diverges from the spec; two fields downstream; fixes the bug by adding a field, not correcting
  the wrong one.

---

## 3. Decision 2 — in-place modification vs. duplication (the cascade)

**The bug [Verified].** Chain: `PostProcessing → SortedSpikes → OfficialCuration →
ApplyOfficialCuration` (each inherits the one above's primary key). `ApplyOfficialCuration.make()`
deletes `SortedSpikes` to swap raw units for curated ones — but that cascades down and deletes its own
`OfficialCuration`/`ApplyOfficialCuration`, so the insert that follows has no parent. Fatal, and
pre-existing in `main`; never caught because the apply step was a draft never run until Zofia tested it.
Her workaround stashes the row and rebuilds in one transaction — it works, but a computed table is
deleting and rebuilding its own ancestor.

**Option 2A — patch in place.** Keep one `SortedSpikes` swapped between raw/curated; fix only the
circularity, e.g. reparent `OfficialCuration` onto `PostProcessing` (same key, since 1:1) so the delete
no longer destroys the curation records.
- Pros: minimal; downstream untouched; unblocks now.
- Cons: treats the symptom — a computed table still deletes/rebuilds an upstream table in place.

**Option 2B — duplication (DataJoint Elements pattern).** Raw `SortedSpikes` becomes immutable; the
curated result is its own downstream table; downstream reads curated, falling back to raw. Nothing is
deleted and rebuilt.
- Pros: removes the circularity and the in-place mutation; matches the Elements pattern
  (`Clustering → Curation → CuratedClustering`).
- Cons: larger restructure; repoints downstream tables; bigger migration.

> Elissa's framing: everything short of 2B still feels like a patch, because the root issue is one
> mutable table holding both raw and curated and swapping in place.

---

## 4. Noise units — keep, label, exclude from matching [Decided — Elissa; confirm with team]

Zofia's branch **deletes** noise-labelled units at apply time. The decision: **keep them, label them
"noise," and exclude them from unit matching** instead. Confirm Zofia/Thinh. [Verified] `main` does
neither — both behaviours are new on the branch. Interacts with Decision 1 (which field the matching
filter reads).

## 5. `unit_quality` sub-choices [Proposal]

If Decision 1 goes single-field: use Kilosort's label as the per-unit fallback for a unit left
unlabelled in a curated block? And mark Kilosort-sourced values with a `ks_` prefix so provenance stays
visible? Both are sub-choices of Decision 1.

## 6. Label-completeness gate [Verified: none exists; Proposal]

There is **no** gate requiring an official curation to have every unit labelled. Decide whether to add
one, or rely on the Kilosort fallback for units the curator didn't touch.

## 7. Applying the changes = repopulation only [Elissa]

These changes recreate/repopulate the affected tables, but this is **not** re-sorting or re-curation:
every sorting analyzer and curation file is saved on disk, so re-applying is only re-populating the
tables — no real work for Zofia. Be explicit about that, and settle with Zofia how she'd like to handle
the re-apply.
