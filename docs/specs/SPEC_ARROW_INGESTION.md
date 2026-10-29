# Arrow-format ephys ingestion

## Why

Bonsai's Aeon acquisition changed how it writes ephys data. Binary wrote
`AmplifierData_n.bin` next to `Clock_n.bin`. Arrow writes one compressed
`<device>_<probe>_n.arrow`, with the amplifier channels, `Clock` and `HubClock` as
columns. The IMU moved the same way and gained `Calibration` and `Temperature`.

`aeon_mecha` reads none of it. Discovery globs `*_AmplifierData*.bin`, finds nothing, and
logs at info level.

## Scope

Both formats, side by side, indefinitely. Arrow is supported from raw files up to
`PreProcessing`, where it becomes a SpikeInterface recording. Everything below that reads
pipeline-produced files and needs no arrow awareness.

Out of scope: the behaviour arm, retiring `probe_assignments.json`, the zarr compression
path for binary, and the Commutator stream.

## The anti-goal

Silent wrong data — shifted spike times, mislabelled channels, a chunk counted twice, an
IMU frame padded with NaN. Every format branch fails loudly on a shape it doesn't
recognise. A crash beats a plausible-looking number.

---

## Design decisions

**Format comes from the file extension.** One `EphysChunk.File` row per file; readers
dispatch on the suffix. A `stream_type` column would query better, but it costs a schema
change and the filename already states the format.

**The same chunk in both formats is an error.** Mixing formats across chunk indices is
fine. The same index in both is not: loading prefers arrow on file existence alone, so
the binary copy would be dropped after its clock had already placed the chunk. The check
asks which files exist, not which ones overlap the window being ingested.

**Metadata decides probe identity**, replacing two guesses:

| Fact | Was | Now |
|---|---|---|
| probe enabled | inferred from a null config filename | explicit `Enable` flag |
| insertion number | assigned at insert time | `subject.Implants[]` |
| subject ↔ probe | `probe_assignments.json` — external, keyed by serial | `subject.Implants[]`, in the epoch's own metadata |

Keying on probe label is safe because the same file maps label to serial, per epoch.
Stale mappings threaten *external* files like `probe_assignments.json`, which stays for
epochs carrying no `Implants` block. Where metadata and database disagree, raise.

One subject per ephys epoch is the current experimental design, not a permanent rule.
Read `Implants[]` under the single subject, but keep that assumption out of the data
model — `ProbeInsertion` already allows a subject per probe.

**The IMU keeps both column sets and records which it has** — 15 columns from arrow, 13
from binary. Each row stores the format it came from, so a fetch returns what was stored.
The flag lives in the row rather than being re-read from disk; otherwise archiving raw
files would change a stored row's shape.

**Schema changes are additive.** The new IMU fields are nullable columns, and nothing is
renamed. DataJoint 2.3.5 makes renaming safe, but `euler_x/y/z` are a published fetch
contract and the column comments already record the axes.

---

## Knowledge not to re-derive

**The IMU column mapping came from the data, not a datasheet.** The arrow names don't
line up with the binary ones, and the obvious reading is wrong:

| Arrow column | Maps to | Evidence |
|---|---|---|
| `EulerAngle_Yaw` | `euler_x` | range [0, 359.94] matches exactly |
| `EulerAngle_Pitch` | `euler_y` | ±90 clamp matches |
| `EulerAngle_Roll` | `euler_z` | ±180 matches |
| `Gravity_{X,Y,Z}` | `gravity_vector_{x,y,z}` | \|Gravity\| = 9.799 ≈ 9.81 |
| `Acceleration_{X,Y,Z}` | `linear_acceleration_{x,y,z}` | \|Accel\| = 4.94, gravity already removed |
| `Quaternion_{X,Y,Z,W}` | `quaternion_{x,y,z,w}` | \|q\| = 1.0000 |

**Streams share an index but not a clock.** IMU chunk N starts 0.03–0.09 s after probe
chunk N, and the two probes sit ~1.9 ms apart. Nothing may assume a shared sample index
across probes.

**Record batches are uniform except the last.** Measured: 600 batches of 30,000 rows,
18,000,000 samples per chunk, the same chunk size binary used. The trace reader addresses
batches by stride, so the code checks this rather than trusting it.

---

## Verification

`golden_arrow_01` joins `abcGolden01` as a golden dataset, and the ephys fixtures run
over both. Real acquisition output covers arrow from discovery through `PreProcessing`,
including the IMU; the synthetic factory covers the `SyncedSpikes` path and trace values.

`PreProcessing` on golden arrow data stops short of writing the preprocessed recording —
the fixture pre-creates it, avoiding 24 GB per block. So the tests cover file selection,
the channel and batch checks against the real schema, concatenation, and the recording
round-trip, but not trace throughput at full scale.

Given up deliberately: no evidence that a real arrow recording sorts sensibly end to end.
Nothing below `PreProcessing` is format-aware, which makes this acceptable — but it is a
gap, not an inference.

Every path that could store wrong data without raising has a test that fails when the fix
is reverted. Each one was verified by reverting it.
