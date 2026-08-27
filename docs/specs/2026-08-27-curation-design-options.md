# Spike-Sorting Curation — Meeting Reference (2026-08-27)

**Purpose:** Talking-points reference for the 2026-08-27 meeting on the spike-sorting curation
pipeline — Zofia's PR **#604** (`zs/spike_sorting`) and the follow-up PR **#608**
(`es/curation-fixes` → `zs/spike_sorting`). Covers what already landed, the two design decisions
that need the team, and the smaller related questions worth raising.

> **How to read this**
> - **[Verified]** — checked against the current branch code, GitHub `main`, and the original spec.
> - **[Proposal]** — Claude's analysis / options for discussion. Not decisions.
> - **[Decided — Elissa]** — a call Elissa has already made (to confirm with the team).
>
> Items under "Other talking points" are candidates to raise, not settled outcomes.

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
5. **`unit_quality` fix details** — If single field: use Kilosort as the per-unit fallback for
   unlabelled units? Mark machine labels with a `ks_` prefix? (Both are sub-choices of Decision 1.)
6. **Duplication (Chris)** — Real duplication points: a full curated-analyzer copy per curation;
   spikes stored twice (SyncedSpikes + UnitMatching.Spikes); leaked blobs on restore; the File table
   mixing the curation JSON with the apply output. **Ask Chris** which he meant.
7. **Backfill** — The label fix is forward-only; blocks Zofia already curated keep the wrong label
   until re-applied. **Decide:** re-apply, migration script, or leave.
8. **Applied-analyzer in `ManualCuration.File`** — Root of the "2 tuples" bug; move it to its own
   table (part of Decision 2)? Restore-cleanup already decided: leave as-is.
9. **Label-completeness gate** — None exists today. **Decide:** add one, or rely on the fallback.
10. **Testing** — Zero automated matching coverage (golden dataset is one block). Zofia tests by hand;
    **build multi-block fixtures?**
11. **Migration + sensitivity** — #608 already recreates tables; a restructure is bigger. **How much
    re-curation of Zofia's existing data is acceptable?**
12. **Ownership** — Cascade/core = Thinh; curation/tags/QC = Zofia; the exception-handler we removed
    was Adrian's, not Zofia's.
13. **DJ bump** — Branch bumps DataJoint `2.2.2 → 2.3.2`; it rides along into main.

*(Full detail on each item is below if you need to drill in.)*

---

## 1. Status — what already landed in PR #608 (partial set) [Verified]

PR #608 collects the safe, agreed fixes so they don't block on the bigger design questions. All go
into `zs/spike_sorting` before it reaches `main`:

- Removed the obsolete uncaught-exception workarounds (DataJoint 2.3.2 fixes it upstream; the branch
  already pins `datajoint>=2.3.2`).
- Fixed "2 tuples found" — name the file in the remaining `ManualCuration.File` reads.
- Made `save_manual_curation`'s two inserts atomic (it's a helper, not a table `make()`).
- Replaced `UnitMatching.CandidateMatch` with `UnitMatching.BlockComparison` — the full
  agreement-score grid per compared block, no threshold.
- Parameterized the curation tags into a `CurationTag` lookup table.

Deferred to this meeting: the quality-label change (Decision 1) and the cascade restructure
(Decision 2).

---

## 2. Decision 1 — how to represent quality labels

### The bug it fixes [Verified]
`SortedSpikes.Unit.unit_quality` is read from Kilosort's `KSLabel` in every code path that builds
the rows, never from the curator's manual `quality` property — so it always reflects Kilosort, never
the human's curation, even after a curation is applied. Pre-existing in `main`. The original spec
(`SPEC_SPIKE_SORTING_CURATION.md`, unchanged since 2026-04-20) intended a **single** quality field:
Kilosort's label before curation, replaced by the curator's label after. On the branch, Zofia added a
**second** column, `curation_quality`, to hold the manual label separately (plus a `UnitTag` part
table for tags); the spec was never updated to describe either.

### Option 1A — single field (the original spec) [Proposal]
One `unit_quality`: the curator's manual label once curated, Kilosort's standing in until then.
- **Pros:** matches the spec; one source of truth; simplest; drops Zofia's extra column (toward `main`).
- **Cons:** Kilosort's label is lost once curated (no side-by-side); a bare label is ambiguous about
  human vs. machine unless marked.
- **Sub-choice:** mark Kilosort-sourced values `ks_good`/`ks_mua`/`ks_noise`, or leave bare.

### Option 1B — two fields (the branch's current state) [Proposal]
`unit_quality` = Kilosort always; `curation_quality` = manual (null until curated).
- **Pros:** both labels visible; no ambiguity; nothing lost.
- **Cons:** diverges from the spec; two fields downstream; "fixes" the bug by adding a field rather
  than correcting the wrong one.

---

## 3. Decision 2 — in-place modification vs. duplication (the cascade)

### The bug it fixes [Verified]
Dependency chain: `PostProcessing → SortedSpikes → OfficialCuration → ApplyOfficialCuration` (each
inherits its primary key from the one above). `ApplyOfficialCuration.make()` deletes the
`SortedSpikes` row to swap raw units for curated ones — but that delete cascades down and removes the
`OfficialCuration` child and the `ApplyOfficialCuration` grandchild, so the next line, which inserts
the `ApplyOfficialCuration` record, has no parent to attach to. Foreign-key violation; the step can't
complete. **Fatal and pre-existing in `main`** — never caught because the apply step was a draft never
run until Zofia tested it. (The spec's "Circular Dependency Handling" section only covers the Python
import cycle, not this table cascade.) Zofia's workaround stashes the `OfficialCuration` row before the
delete, rebuilds `SortedSpikes` from the in-memory analyzer, and re-inserts both in one transaction —
it works, but a computed table is reaching up to delete and rebuild its own ancestor.

### Option 2A — patch in place [Proposal]
Keep one `SortedSpikes` table swapped between raw and curated; fix only the circularity.
- **Variant i — reparent:** move `OfficialCuration`'s parent from `SortedSpikes` up to
  `PostProcessing` (same primary key, since they're 1:1). Deleting `SortedSpikes` no longer destroys
  the curation records, and Zofia's in-memory recreate can be removed.
- **Variant ii — keep Zofia's in-memory recreate** as the workaround.
- **Pros:** minimal; downstream untouched; unblocks now.
- **Cons:** treats the symptom, not the cause — a computed table still deletes/rebuilds an upstream
  table in place; the reparent also loosens the enforced link between "which curation is official" and
  "what's in `SortedSpikes` now" (left to the `curation_id` convention); still a schema change.

### Option 2B — duplication (DataJoint Elements pattern) [Proposal]
Raw `SortedSpikes` becomes immutable; the curated result lives in its own downstream table computed
from `OfficialCuration`; downstream reads the curated table, falling back to raw when there's no
curation. Nothing is ever deleted and rebuilt.
- **Pros:** removes both the circularity and the in-place mutation; matches the established DataJoint
  Elements curation pattern (`Clustering → Curation → CuratedClustering`); clean lineage.
- **Cons:** larger restructure; repoints downstream tables (`SyncedSpikes`, `UnitMatching`, …); a
  second units table with some duplicated extraction logic; migration.

> Note (Elissa's framing): every option short of 2B still feels like a patch, because the root issue
> is one mutable table holding both the raw and curated result and swapping in place.

---

## Other talking points

### 4. Noise units — keep, label, exclude from matching [Decided — Elissa; confirm with team]
Zofia's branch currently **deletes** noise-labelled units at apply time. The decision is to **keep
them, label them "noise," and exclude them from unit matching** instead of deleting. Confirm Zofia and
Thinh agree. Interacts with Decision 1 (which field the matching filter reads). [Verified] `main` does
neither — it never deletes noise and never excludes it from matching; both behaviours are new on the
branch.

### 5. The `unit_quality` fix — fallback, prefix, keep-both [Proposal]
If Decision 1 goes single-field, open sub-questions: the per-unit fallback for a unit left unlabelled
in a curated block (use Kilosort's label as a stand-in?); whether to mark Kilosort-sourced values with
a `ks_` prefix so provenance is visible; and the standing argument for keeping both the Kilosort and
manual labels (which is really Decision 1 itself).

### 6. Data duplication — what's actually duplicated [Verified] (Chris's concern)
Worth pinning down which of these Chris meant:
- The curated analyzer is a **full second copy on disk** per applied curation; restore deletes neither
  the folder nor its `File` row.
- Spike payloads are stored **twice** — `SyncedSpikes.Unit.spike_times`, then again in
  `UnitMatching.Spikes.spike_times` keyed by `global_unit`.
- Apply/restore cycles **leak old external blobs** (DataJoint's row delete doesn't remove
  external-store files).
- `ManualCuration.File` mixes two different kinds of thing (the curation JSON and the apply-step's
  analyzer pointer) — the root of the "2 tuples" bug. Connects to Decision 2.

### 7. Backfilling already-curated blocks [Proposal]
The `unit_quality` fix is forward-looking: blocks Zofia has already curated keep the buggy Kilosort
label until re-applied. Manual labels are safe on disk in the curated analyzers, so nothing is lost,
but correcting existing blocks means re-applying each curation (disruptive) or a one-off migration
script. Decide whether and how to backfill.

### 8. Applied-analyzer in `ManualCuration.File` + restore cleanup [Verified; Decided — Elissa on cleanup]
The apply step stores the curated-analyzer path in `ManualCuration.File` — an apply-step output living
in a manual-curation file table, which is what makes the "2 tuples" collision possible. Whether it
should move to its own table is part of Decision 2. Separately, [Decided — Elissa] leave
`restore_raw_sorting` as-is: with the file-name fix, the leftover applied-analyzer rows are harmless
and are useful provenance; the only cost is on-disk folders accumulating.

### 9. Label-completeness gate [Verified: none exists; Proposal]
There is **no** gate requiring an official curation to have every unit labelled. Decide whether to add
one, or rely on the Kilosort per-unit fallback for units the curator didn't touch.

### 10. Testing gap [Verified]
There is **no automated coverage** for unit matching — the golden dataset is a single block, and
matching needs several overlapping curated blocks. Zofia is asked (in PR #608) to test the matching
changes on real data. [Proposal] consider building multi-block test fixtures so this path isn't
verified only by hand.

### 11. Schema migration + sensitivity [Verified / Elissa]
PR #608 already recreates tables on the live DB (adds `CurationTag` and `BlockComparison`, makes
`UnitTag.tag` a foreign key, removes `CandidateMatch`). Decision 2 (restructure) would be a bigger
migration. [Elissa] Zofia is running on real curated data and won't want to re-curate, so major
changes should be weighed against how much of her existing work they'd force her to redo.

### 12. Ownership / who decides [Proposal]
- Cascade + core table structure (Decision 2) — **Thinh** (shared/core design).
- Curation workflow, tags, QC — **Zofia** (her PR).
- The removed `__init__.py` exception-handler workaround was **Adrian's** (via `ar/core`), not Zofia's.

### 13. DataJoint version bump [Verified]
The branch bumps DataJoint from `main`'s `>=2.2.2` to `>=2.3.2` (needed for the upstream traceback
fix). Flag that this rides along with the branch into `main`.
