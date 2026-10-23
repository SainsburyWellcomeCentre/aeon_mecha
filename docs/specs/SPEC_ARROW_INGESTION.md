# Arrow-format ephys ingestion

Status: approved 2026-10-21 · Issue [#634](https://github.com/SainsburyWellcomeCentre/aeon_mecha/issues/634)

## Why

Bonsai's Aeon acquisition changed how it writes ephys data. Probe, clock and IMU streams
are now Arrow IPC files, compressed as they're written, with several streams sharing one
file. Where binary wrote `AmplifierData_n.bin` beside `Clock_n.bin`, arrow writes one
`<device>_<probe>_n.arrow` holding the amplifier channels, `Clock` and `HubClock` as
columns. The IMU moved the same way and gained two fields it never had
(`Calibration`, `Temperature`).

`aeon_mecha` can't ingest any of it — discovery globs `*_AmplifierData*.bin`, finds
nothing, and logs at info level.

## Scope

Both formats, side by side, indefinitely. Nothing is removed, so this is additive.

Arrow is supported from raw files through to `PreProcessing`, where it becomes a
SpikeInterface recording. Everything below reads pipeline-produced intermediates and
needs no arrow awareness.

**Out of scope:** the behaviour arm (its metadata is unchanged); removing
`probe_assignments.json`; the zarr compression path for binary data; and Commutator
ingestion, deferred below.

## The anti-goal

**Silent wrong data** — shifted spike times, mislabelled channels, a chunk counted
twice, an IMU frame padded with NaN. Every format branch fails loudly on a shape it
doesn't recognise. A loud failure is a good outcome here; a plausible-looking number is
not.

---

## Design decisions

**Format comes from the file extension, decided in code.** `EphysChunk.File` keeps one
row per physical file and consumers dispatch on the suffix. The alternative — a
`stream_type` column — is more queryable but is a schema change needing sign-off for
something the filename already states unambiguously. Revisit if a format appears whose
extension doesn't determine its layout.

**A chunk present in both formats is an error.** An epoch written across a format change
may mix formats across different chunk indices, but the same index in both is ambiguous:
loading prefers arrow on file existence alone, so the binary copy would be dropped after
its clock had already decided where the chunk belongs. The check is on which files
exist, not on which ones overlap the window being ingested.

**Metadata is the authority for probe identity.** Two inference sites are removed:

| Fact | Was | Now |
|---|---|---|
| probe enabled | inferred from a null config filename | explicit `Enable` flag |
| insertion number | derived at insert time | `subject.Implants[]` |
| subject ↔ probe | `probe_assignments.json`, external, serial-keyed | `subject.Implants[]`, in the epoch's own metadata |

A label-keyed mapping is safe because the same file defines label→serial and is written
per epoch; the staleness failure mode needs an *external* mapping, which is what
`probe_assignments.json` is. That file stays for epochs with no `Implants` block —
retiring it is a separate decision, and it's the escape hatch if multi-subject epochs
return before the metadata can express them.

Where metadata and database disagree, raise. Picking a winner silently is the anti-goal
wearing a different hat.

**The IMU keeps both column sets and records which it has.** Arrow chunks carry 15
columns, binary 13. Rather than normalising away `Calibration` and `Temperature`, each
row records the format it was ingested from, so a fetch returns what was stored. That
discriminator lives in the row, not re-derived from disk at fetch time, or archiving the
raw files would change a stored row's shape.

**Schema changes are additive only.** Not a style preference: DataJoint 2.3.2's
attribute-rename marker is dead code. Tested against live MySQL 8.0 across three types,
the generated DDL is `DROP` + `ADD` rather than `CHANGE`, so the data is destroyed
without an error — detection is a regex against the column comment, which DataJoint
itself defeats by prefixing every comment with its type. Any rename shipped here would
be a landmine for whoever next runs `alter()`. Pure `ADD` is verified safe. *This is an
upstream DataJoint bug and should be reported.*

---

## Knowledge not to re-derive

**The IMU column mapping came from the data, not the datasheet.** The arrow names don't
line up with the binary ones and the obvious reading is wrong:

| Arrow column | Maps to | Evidence |
|---|---|---|
| `EulerAngle_Yaw` | `euler_x` | range [0, 359.94] matches exactly |
| `EulerAngle_Pitch` | `euler_y` | ±90 clamp matches |
| `EulerAngle_Roll` | `euler_z` | ±180 matches |
| `Gravity_{X,Y,Z}` | `gravity_vector_{x,y,z}` | \|Gravity\| = 9.799 ≈ 9.81 |
| `Acceleration_{X,Y,Z}` | `linear_acceleration_{x,y,z}` | \|Accel\| = 4.94, so gravity is already removed |
| `Quaternion_{X,Y,Z,W}` | `quaternion_{x,y,z,w}` | \|q\| = 1.0000 |

**Streams are index-aligned but not clock-aligned.** IMU chunk N starts 0.03–0.09 s after
probe chunk N, and the two probes are ~1.9 ms apart. Overlap logic stays necessary, and
nothing may assume a shared sample index across probes.

**Record batches are uniform except the last in each file.** The trace-reading path
depends on that, so it is checked rather than assumed. Chunks still hold the same
18,000,000 samples binary did.

---

## Verification

Golden datasets are added, not swapped: `golden_arrow_01` joins `abcGolden01` and the
ephys fixtures run over both.

| Covered by | What |
|---|---|
| `golden_arrow_01` | ingestion through `EphysChunk`, plus IMU, on real acquisition output |
| synthetic factory, arrow variant | the `PreProcessing` and `SyncedSpikes` arrow paths |
| `abcGolden01` | unchanged — the full cascade through curation and unit matching |

`PreProcessing` is keyed on `SortingTask`, so reaching it means giving a dataset an
electrode group and a parameter set — sorting configuration, not ingestion
configuration. Arrow's coverage there is synthetic rather than golden, which would need
a 109 GB intermediate.

**Given up deliberately:** no evidence that a real arrow recording sorts sensibly end to
end. Acceptable because there's no arrow-specific logic in the sorter or below, but
stated rather than implied.

Anything that could put wrong data in the database without raising is covered by a test
that fails when the fix is reverted — verified by reverting it.

---

## Deferred: Commutator

Present in both golden datasets and never ingested — a pre-existing gap, not arrow work.

**Blocked on data, not design.** Every `Turns` sample is zero in both recordings: the
clock is real and monotonic, so the stream records correctly, but the commutator never
rotated. All-zero bytes are identical under `uint32`, `int32` and `float32`, so the dtype
can't be read from the data, and the things most likely to be wrong — sign convention,
whether turns accumulate or reset per chunk, scaling — are invisible in zeros.

When a recording with real rotation exists: a separate table mirroring `OnixImuChunk`,
keyed on `EphysSyncModel`, samples stored in-database. Separate rather than columns on
the IMU table because the rates differ (10 Hz against 100 Hz) on different chunk
boundaries. Confirm the dtype from the acquisition schema, not by inspection.

---

## Known loss of a correctness guard

The acquisition fixed a quirk where Bonsai added the HARP sync second twice into the
`Seconds` column. Production was always correct — the sync model fits on
`Value.HarpTime` — but one test proved it by asserting the `Seconds`-derived value was
*absent* from stored sync starts. On new-format data the two columns are identical, so
that assertion can't discriminate and would fail against a correct reader. It now runs
only when the columns differ.

**So on new-format acquisitions nothing would catch the reader switching to `Seconds`.**
The guard survives only for older data.

---

## Constraints from the team

- **One subject per ephys epoch is the current experimental design, not a permanent
  constraint.** Read `Implants[]` under the single subject, but don't bake that into the
  data model — `ProbeInsertion` already expresses per-probe subjects.
- **Commutator data with real rotation exists**, we just don't have it locally. The
  deferral above is an access problem, not a data-existence one.
