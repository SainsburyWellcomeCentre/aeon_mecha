# Arrow-format ephys ingestion

Status: approved 2026-10-21 · Issue [#634](https://github.com/SainsburyWellcomeCentre/aeon_mecha/issues/634)

## Why

Bonsai's Aeon acquisition changed how it writes ephys data. Probe, clock and IMU streams
are now Arrow IPC files, compressed as they're written, with several streams sharing one
file. Where binary wrote `AmplifierData_n.bin` beside `Clock_n.bin`, arrow writes one
`<device>_<probe>_n.arrow` holding the amplifier channels, `Clock` and `HubClock` as
columns. The IMU moved the same way and gained two fields it never had before
(`Calibration`, `Temperature`).

`aeon_mecha` can't ingest any of it — discovery globs `*_AmplifierData*.bin`, finds
nothing, and logs at info level.

## Scope

Both formats, side by side, indefinitely. Nothing is removed, so this is additive and a
minor version.

Arrow is supported from raw files through to `PreProcessing`, where arrow data becomes a
SpikeInterface recording. Everything below that point reads pipeline-produced
intermediates and never touches raw files, so it needs no arrow awareness.

**Out of scope:** the behaviour arm (its metadata is structurally unchanged); removing
`probe_assignments.json`; reworking the zarr compression path for binary data; and
Commutator ingestion, which is deferred for want of data to validate against.

## The anti-goal

**Silent wrong data.** Shifted spike times, mislabelled channels, a chunk counted twice,
an IMU frame padded with NaN. Every format branch fails loudly on a shape it doesn't
recognise. A loud failure is a good outcome here; a plausible-looking number is not.

The current behaviour — a glob matching nothing and ingestion returning quietly — is the
failure mode being removed, not a precedent to follow.

---

## Design decisions

### Format is determined by file extension, in code

`EphysChunk.File` keeps one row per physical file. Consumers that need a particular
stream fetch the rows and dispatch on the suffix: arrow reads the named column, binary
resolves its companion file.

*Rejected:* adding `stream_type` / `file_format` columns. More queryable, but it's a
schema change needing sign-off for information the filename already carries
unambiguously. Worth revisiting if a format appears where the extension doesn't determine
the layout.

This is not the inference being removed below — an extension states the physical format,
where that inference guessed *semantics* from a *missing* field.

### A chunk present in both formats is an error

An epoch written across a format change may mix formats across different chunk indices.
The same chunk index in both formats is refused, on both the amplifier and IMU sides,
because which copy is authoritative would be a guess — and loading picks arrow on file
existence alone, so the binary copy would be dropped after its clock had already decided
where the chunk belongs.

Ambiguity is decided on which files exist, not on which ones overlap the window being
ingested. The latter catches only the subset that happens to be visible from one query.

### Metadata becomes the authority for probe identity

Two inference sites are removed:

| Fact | Was | Now |
|---|---|---|
| probe enabled | inferred from a null config filename | explicit `Enable` flag |
| insertion number | derived at insert time | `subject.Implants[].InsertionNumber` |
| subject ↔ probe | `probe_assignments.json`, external, serial-keyed | `subject.Implants[]`, in the epoch's own metadata |

A label-keyed mapping is safe here because the same file also defines label→serial and is
regenerated per epoch. The staleness failure mode needs an *external* mapping, which is
what `probe_assignments.json` is.

If metadata and database disagree about an insertion number, raise. Don't reconcile —
picking a winner silently is the anti-goal wearing a different hat.

`probe_assignments.json` stays for epochs with no `Implants` block. Retiring it is a
separate, deliberate decision, and it's still the escape hatch if multi-subject epochs
return before the metadata can express them.

### The IMU stores both column sets, and records which it has

Arrow IMU chunks carry 15 columns, binary 13. Rather than normalising and discarding
`Calibration` and `Temperature`, each row records the format it was ingested from, so a
fetch returns what was actually stored.

That discriminator lives in the row rather than being re-derived from disk at fetch time.
Otherwise archiving or converting raw files silently changes a stored row's shape.

The format is a property of the epoch, not of whether a particular window held samples —
an empty row carries the same columns as its populated siblings.

### Schema changes are additive only

The two new IMU fields are added as nullable columns; nothing is renamed.

This isn't a style preference. **DataJoint 2.3.2's attribute-rename marker is dead code.**
Tested against live MySQL 8.0 across three types: the generated DDL is `DROP` + `ADD`, not
`CHANGE`, so the data is destroyed without an error. Detection is a regex against the
column comment, which DataJoint itself defeats by prefixing every comment with its type.

So any rename shipped here would be a landmine for whoever next runs `alter()`. Pure
`ADD` is verified safe. Explicit DDL can rename and preserve data if it's ever needed.
*This is an upstream DataJoint bug and should be reported.*

**Recorded alternative:** a `Column` part table keyed on column name would make the column
set queryable per row and end the DDL churn — a fixed column list fundamentally can't
describe two formats with different column sets. Deferred because per-column JSON is the
house style and two nullable columns is the smaller change. Revisit if a third IMU column
set appears, rather than adding more nullable columns.

---

## Knowledge not to re-derive

### The IMU column mapping came from the data, not the datasheet

The arrow names don't line up with the binary ones, and the obvious reading would be
wrong. Verified against real recordings:

| Arrow column | Maps to | Evidence |
|---|---|---|
| `EulerAngle_Yaw` | `euler_x` | range [0, 359.94] matches exactly |
| `EulerAngle_Pitch` | `euler_y` | ±90 clamp matches |
| `EulerAngle_Roll` | `euler_z` | ±180 matches |
| `Gravity_{X,Y,Z}` | `gravity_vector_{x,y,z}` | \|Gravity\| = 9.799 ≈ 9.81 |
| `Acceleration_{X,Y,Z}` | `linear_acceleration_{x,y,z}` | \|Accel\| = 4.94, so gravity is already removed |
| `Quaternion_{X,Y,Z,W}` | `quaternion_{x,y,z,w}` | \|q\| = 1.0000 |
| `Calibration` | new | BNO055 status byte |
| `Temperature` | new | constant over the sampled window |

### Streams are index-aligned but not clock-aligned

IMU chunk N starts 0.03–0.09 s after probe chunk N, and the two probes are ~1.9 ms apart.
Overlap logic stays necessary, and nothing may assume a shared sample index across
probes.

### The chunk convention didn't change

Arrow chunks hold the same 18,000,000 samples binary did. Record batches are uniform
except the last in each file — a property the trace-reading path depends on, so it's
checked rather than assumed.

---

## Verification strategy

Golden datasets are added, not swapped. `golden_arrow_01` joins `abcGolden01` and the
ephys fixtures run over both, so existing coverage is untouched.

| Covered by | What |
|---|---|
| `golden_arrow_01` | ingestion through `EphysChunk`, plus IMU, on real acquisition output |
| synthetic factory, arrow variant | the `PreProcessing` and `SyncedSpikes` arrow paths |
| `abcGolden01` | unchanged — the full cascade through curation and unit matching |

`PreProcessing` is keyed on `SortingTask`, so it sits *inside* the sorting cascade rather
than above it. Reaching it for a dataset means giving that dataset an electrode group and
a parameter set — sorting configuration, not ingestion configuration. Arrow's coverage
there comes from the synthetic factory, which builds a sorting task cheaply, rather than
from the golden dataset, which would need a 109 GB intermediate.

**Given up deliberately:** no evidence that a real arrow recording sorts sensibly end to
end. Acceptable because there's no arrow-specific logic in the sorter or below, but it's
stated rather than implied. A real arrow golden sorting stays additive if wanted later.

Anything that could put wrong data in the database without raising is covered by a test
that fails when the fix is reverted — verified by reverting, not by reading the diff.

---

## Deferred: Commutator

Present in both golden datasets and never ingested — a pre-existing gap, not arrow work.

**Blocked on data, not design.** Every `Turns` sample is zero in both recordings. The
clock is real and monotonic, so the stream records correctly; the commutator simply never
rotated. That makes two things impossible:

1. **The dtype can't be read from the data** — all-zero bytes are identical under
   `uint32`, `int32` and `float32`. It has to come from the acquisition schema.
2. **There's nothing to validate against.** The things most likely to be wrong — dtype,
   sign convention, whether turns accumulate or reset per chunk, scaling — are all
   invisible in zeros.

**Recorded design:** a separate table mirroring `OnixImuChunk`, keyed on `EphysSyncModel`,
with the samples stored in-database as `<blob>`. Separate rather than columns on the IMU
table because the rates differ (10 Hz against 100 Hz) on different chunk boundaries, which
would force resampling or NaN padding. In-database because the reference codec rebuilds
from raw files on fetch, and for ~216 KB per row self-containment is worth more than the
saving.

Before picking this up: find a recording where the commutator turned, and confirm the
dtype from the acquisition schema.

---

## Known loss of a correctness guard

The acquisition fixed a quirk where Bonsai added the HARP sync second twice into the
`Seconds` index column. Production was always correct — the sync model fits on
`Value.HarpTime`, never on `Seconds` — but one test proved it by asserting the
`Seconds`-derived value was *absent* from stored sync starts.

On new-format data the two columns are identical, so that assertion can't discriminate at
all; it would fail against a perfectly correct reader. It now runs only when the columns
differ.

**So on new-format acquisitions there's no longer a test that would catch the reader
switching to the `Seconds` column.** The guard survives only for older data. Driven by an
acquisition change this spec otherwise declares out of scope, and recorded here rather
than buried in a test comment.

---

## Constraints from the team

- **One subject per ephys epoch is the current experimental design, not a permanent
  constraint.** Read `Implants[]` under the single subject, but don't bake the assumption
  into the data model — `ProbeInsertion` already expresses per-probe subjects and must
  keep doing so.
- **Commutator data with real rotation exists**, we just don't have it locally. The
  deferral above is an access problem, not a data-existence one.
