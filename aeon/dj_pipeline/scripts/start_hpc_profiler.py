#!/usr/bin/env python3
"""Periodic resource logger scoped to *this job*, not the whole machine.

Compute nodes are shared, so node-wide counters say nothing about what our own
work costs. Where possible every metric is scoped to the SLURM job and reported
as a percentage of the *allocation* (what `--mem`/`--cpus-per-task`/`--gres`
asked for) rather than the node's specs.

Scoping comes from the SLURM cgroup (v2) that stepd puts the job in: the job's
own `memory.current` / `cpu.stat`, and the limits SLURM enforces on them. Off
SLURM (login node, laptop) it falls back to summing this user's processes and
comparing against the machine's totals; the `scope` header records which was
used.

The allocation and node details are written as `#` comment lines above the CSV
header, so a log stays interpretable long after the job is gone.

Usage:
    python start_hpc_profiler.py -o resource_use.csv & PROFILER_PID=$!
    # Run the script to profile here
    kill $PROFILER_PID

Adrian 2025-09-29
"""

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

CGROUP_ROOT = Path("/sys/fs/cgroup")
GB = 1e-9
MB = 1e-6


# --------------------------------------------------------------------------- #
# cgroup helpers
# --------------------------------------------------------------------------- #

def find_job_cgroup():
    """Return the SLURM job-level cgroup path, or None if not in one.

    /proc/self/cgroup holds a path like
        0::/system.slice/<node>_slurmstepd.scope/job_1234/step_0/user/task_0
    We truncate at the `job_<id>` component so the numbers cover every step of
    the job (including anything srun spawns later), not just our own step.
    """
    try:
        content = Path("/proc/self/cgroup").read_text()
    except OSError:
        return None

    for line in content.splitlines():
        hid, _, path = line.partition(":")
        if hid != "0":  # only the cgroup-v2 unified hierarchy carries the path
            continue
        parts = path.lstrip(":").strip("/").split("/")
        for i, part in enumerate(parts):
            if part.startswith("job_"):
                cgroup = CGROUP_ROOT.joinpath(*parts[: i + 1])
                return cgroup if cgroup.is_dir() else None
    return None


def read_int(path, default=None):
    """Read a single-integer cgroup file. 'max' (no limit) yields `default`."""
    try:
        text = path.read_text().strip()
    except OSError:
        return default
    return default if text == "max" else int(text)


def read_keyed(path):
    """Parse a 'key value' cgroup file (memory.stat, cpu.stat) into a dict."""
    stats = {}
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition(" ")
            stats[key] = int(value)
    except (OSError, ValueError):
        pass
    return stats


def read_pressure(path, kind="some", window="avg10"):
    """Read a PSI stall percentage from a cgroup {cpu,memory}.pressure file.

    Lines look like:
        some avg10=0.00 avg60=0.01 avg300=0.09 total=2224794
    'some' is the share of wall time at least one task was stalled waiting for
    the resource. This is the honest "is the limit hurting me" signal: usage can
    sit pinned at 100% of the limit with zero stall (see _memory).
    """
    try:
        for line in path.read_text().splitlines():
            if line.startswith(kind):
                match = re.search(rf"{window}=([\d.]+)", line)
                if match:
                    return float(match.group(1))
    except OSError:
        pass
    return 0.0


def count_cpu_list(text):
    """Count CPUs in a cpuset list such as '24,152' or '0-3,8' -> 2, 5."""
    total = 0
    for part in text.split(","):
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        total += int(hi) - int(lo) + 1 if sep else 1
    return total


def cgroup_pids(cgroup):
    """PIDs in `cgroup` and its descendants.

    cgroup v2 only holds processes in leaf nodes, so the job-level cgroup.procs
    is empty and every child has to be walked.
    """
    pids = set()
    for procs in cgroup.rglob("cgroup.procs"):
        try:
            pids.update(int(p) for p in procs.read_text().split())
        except (OSError, ValueError):
            continue
    return pids


# --------------------------------------------------------------------------- #
# nvidia-smi helpers
# --------------------------------------------------------------------------- #

def nvidia_smi(query, fields):
    """Run an nvidia-smi CSV query, returning a list of per-row field lists.

    Returns [] when there is no GPU, no driver, or none allocated to us --
    nvidia-smi exits 6 ("No devices were found") in that last case, because
    SLURM's ConstrainDevices hides GPUs the job does not own.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-{query}={fields}",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return [[f.strip() for f in line.split(",")]
            for line in out.strip().splitlines() if line.strip()]


def to_float(text, default=0.0):
    """Parse an nvidia-smi number, tolerating '[N/A]' and blanks."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #

class Allocation:
    """What this job may use, and how to measure what it does use.

    Limits come from the cgroup where possible (the value the kernel actually
    enforces) and from SLURM_* env vars otherwise.
    """

    def __init__(self):
        self.cgroup = find_job_cgroup()
        self.scope = "slurm-cgroup" if self.cgroup else "user-processes"
        self.uid = os.getuid()

        self.n_cpus = self._cpu_limit()
        self.mem_limit = self._mem_limit()
        self.gpus = nvidia_smi("gpu", "index,name,memory.total,driver_version")

    def _cpu_limit(self):
        """Allocated core count.

        SLURM pins jobs with cpuset rather than a cpu.max quota (cpu.max reads
        'max' here), so the cpuset size is the real limit.
        """
        if self.cgroup:
            try:
                return count_cpu_list(
                    (self.cgroup / "cpuset.cpus.effective").read_text().strip())
            except OSError:
                pass
        for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            if os.environ.get(var):
                return int(os.environ[var])
        # len(sched_getaffinity) respects a cpuset even without cgroup access.
        return len(os.sched_getaffinity(0)) or psutil.cpu_count()

    def _mem_limit(self):
        """Allocated memory in bytes."""
        if self.cgroup:
            limit = read_int(self.cgroup / "memory.max")
            if limit is not None:
                return limit
        for var, scale in (("SLURM_MEM_PER_NODE", 1), ("SLURM_MEM_PER_CPU", self.n_cpus)):
            if os.environ.get(var):
                return int(os.environ[var]) * scale * 1024**2  # SLURM reports MB
        return psutil.virtual_memory().total

    def pids(self):
        """PIDs to attribute usage to."""
        if self.cgroup:
            return cgroup_pids(self.cgroup)
        return {p.pid for p in psutil.process_iter(["uids"])
                if p.info["uids"] and p.info["uids"].real == self.uid}


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #

class Sampler:
    """Turns cumulative counters into per-interval rates."""

    def __init__(self, alloc):
        self.alloc = alloc
        self.start = time.monotonic()
        self.prev_time = None
        self.prev_cpu = {}
        self.prev_io = {}

    def _cpu(self, elapsed):
        """(total, user, system) cores busy over the last interval."""
        if self.alloc.cgroup:
            stat = read_keyed(self.alloc.cgroup / "cpu.stat")
            now = {k: stat.get(k, 0) * 1e-6
                   for k in ("usage_usec", "user_usec", "system_usec")}
        else:
            total = user = system = 0.0
            for proc in psutil.process_iter(["uids", "cpu_times"]):
                info = proc.info
                if not info["uids"] or info["uids"].real != self.alloc.uid:
                    continue
                times = info["cpu_times"]
                if times:
                    user += times.user
                    system += times.system
                    total += times.user + times.system
            now = {"usage_usec": total, "user_usec": user, "system_usec": system}

        prev, self.prev_cpu = self.prev_cpu, now
        if not prev or not elapsed:
            return 0.0, 0.0, 0.0
        return tuple(max(now[k] - prev.get(k, 0.0), 0.0) / elapsed
                     for k in ("usage_usec", "user_usec", "system_usec"))

    def _memory(self):
        """(current, anon, cache, shmem, peak) bytes charged to the job.

        `current` is what SLURM's limit applies to, but it includes page cache.
        A job mmap-ing a big recording off /ceph parks the whole file there, so
        `current` pins to the limit while the job holds little of its own -- the
        kernel just reclaims clean pages instead of OOM-killing.

        The split that predicts an OOM kill is anon + shmem, not `current`:
        those cannot be reclaimed (this cluster sets swap.max=0), while file
        cache can. shmem is broken out because memory.stat counts /dev/shm under
        `file` even though it is unreclaimable -- and SpikeInterface passes
        recordings between workers through /dev/shm, so the difference is the
        difference between a harmless cache and an OOM kill.
        """
        if self.alloc.cgroup:
            stat = read_keyed(self.alloc.cgroup / "memory.stat")
            current = read_int(self.alloc.cgroup / "memory.current", 0)
            peak = read_int(self.alloc.cgroup / "memory.peak", 0)
            shmem = stat.get("shmem", 0)
            cache = max(stat.get("file", 0) - shmem, 0)  # `file` includes shmem
            return current, stat.get("anon", 0), cache, shmem, peak

        # Fallback: RSS double-counts shared pages, but is the cheap estimate.
        rss = 0
        for proc in psutil.process_iter(["uids", "memory_info"]):
            info = proc.info
            if info["uids"] and info["uids"].real == self.alloc.uid and info["memory_info"]:
                rss += info["memory_info"].rss
        return rss, rss, 0, 0, 0

    def _io(self, pids, elapsed):
        """(read, write) bytes/s across the job's processes.

        rchar/wchar count bytes moved by syscalls, so unlike block-level
        accounting they include reads from /ceph (NFS). Summed per PID as
        deltas: a process that exits between samples drops its final partial
        interval, which beats the whole total collapsing when it disappears.
        """
        now, read_delta, write_delta = {}, 0, 0
        for pid in pids:
            try:
                fields = dict(
                    line.split(": ") for line in
                    Path(f"/proc/{pid}/io").read_text().splitlines() if ": " in line)
                counters = (int(fields["rchar"]), int(fields["wchar"]))
            except (OSError, KeyError, ValueError):
                continue  # process exited, or not ours to read
            now[pid] = counters
            prev = self.prev_io.get(pid)
            if prev:
                read_delta += max(counters[0] - prev[0], 0)
                write_delta += max(counters[1] - prev[1], 0)

        first_pass = not self.prev_io
        self.prev_io = now
        if first_pass or not elapsed:
            return 0.0, 0.0
        return read_delta / elapsed, write_delta / elapsed

    def _gpu(self, pids):
        """Per-GPU stats, with memory summed over *our* processes only.

        nvidia-smi's memory.used is device-wide; when a GPU is shared that
        includes other jobs, so it is recomputed from the compute-apps list.
        Utilisation cannot be attributed per process this way and stays
        device-wide.
        """
        if not self.alloc.gpus:
            return dict.fromkeys(
                ("util", "mem_used", "mem_total", "temp", "power"), 0.0)

        rows = nvidia_smi(
            "gpu", "utilization.gpu,memory.total,temperature.gpu,power.draw")
        util = [to_float(r[0]) for r in rows]
        stats = {
            "util": sum(util) / len(util) if util else 0.0,
            "mem_total": sum(to_float(r[1]) for r in rows) * 1e6,  # MiB -> B
            "temp": max((to_float(r[2]) for r in rows), default=0.0),
            "power": sum(to_float(r[3]) for r in rows),
        }
        stats["mem_used"] = sum(
            to_float(r[1]) * 1e6 for r in nvidia_smi("compute-apps", "pid,used_gpu_memory")
            if r[0].isdigit() and int(r[0]) in pids)
        return stats

    def sample(self):
        """One row of measurements as a list of preformatted strings."""
        now = time.monotonic()
        elapsed = (now - self.prev_time) if self.prev_time else 0.0
        self.prev_time = now

        pids = self.alloc.pids()
        cpu_total, cpu_user, cpu_system = self._cpu(elapsed)
        mem_current, mem_anon, mem_cache, mem_shmem, mem_peak = self._memory()
        io_read, io_write = self._io(pids, elapsed)
        gpu = self._gpu(pids)

        cgroup = self.alloc.cgroup
        events = read_keyed(cgroup / "memory.events") if cgroup else {}
        mem_stall = read_pressure(cgroup / "memory.pressure") if cgroup else 0.0
        cpu_stall = read_pressure(cgroup / "cpu.pressure") if cgroup else 0.0

        mem_limit = self.alloc.mem_limit or 1
        return [
            datetime.now().isoformat(timespec="milliseconds"),
            f"{now - self.start:.1f}",
            f"{cpu_total:.2f}",
            f"{100 * cpu_total / self.alloc.n_cpus:.1f}",
            f"{cpu_user:.2f}",
            f"{cpu_system:.2f}",
            f"{mem_current * GB:.2f}",
            f"{mem_anon * GB:.2f}",
            f"{mem_cache * GB:.2f}",
            f"{mem_shmem * GB:.2f}",
            f"{100 * mem_current / mem_limit:.1f}",
            f"{100 * (mem_anon + mem_shmem) / mem_limit:.1f}",
            f"{mem_peak * GB:.2f}",
            f"{mem_stall:.2f}",
            f"{cpu_stall:.2f}",
            str(events.get("max", 0)),
            str(events.get("oom_kill", 0)),
            f"{io_read * MB:.1f}",
            f"{io_write * MB:.1f}",
            f"{gpu['util']:.1f}",
            f"{gpu['mem_used'] * GB:.2f}",
            f"{gpu['mem_total'] * GB:.2f}",
            f"{gpu['temp']:.0f}",
            f"{gpu['power']:.1f}",
            str(len(pids)),
        ]


COLUMNS = [
    "timestamp",
    "elapsed_s",
    "cpu_cores_used",           # cores busy, averaged over the interval
    "cpu_percent",              # of allocated cores
    "cpu_user_cores",
    "cpu_system_cores",
    "mem_used_GB",              # cgroup memory.current: what --mem is charged for
    "mem_anon_GB",              # actually-held memory; unreclaimable
    "mem_cache_GB",             # file cache (mmap-ed /ceph reads); reclaimable, harmless
    "mem_shmem_GB",             # /dev/shm; unreclaimable (swap is off) despite being "shared"
    "mem_percent",              # of allocated memory; inflated by cache, see mem_cache_GB
    "mem_unreclaimable_percent",  # (anon+shmem) of allocated memory -- this predicts OOM kills
    "mem_peak_GB",              # high-water mark since job start
    "mem_pressure_percent",     # % of time stalled on memory (PSI); 0 => limit is fine
    "cpu_pressure_percent",     # % of time stalled waiting for a core (PSI)
    "mem_limit_hits",           # cumulative: times the cgroup hit its limit and reclaimed
    "mem_oom_kills",            # cumulative: the only number that means --mem was too small
    "io_read_MBps",             # syscall-level, so /ceph (NFS) counts
    "io_write_MBps",
    "gpu_utilization_percent",  # device-wide, not job-scoped
    "gpu_memory_used_GB",       # this job's processes only
    "gpu_memory_total_GB",
    "gpu_temperature_C",
    "gpu_power_draw_W",
    "n_processes",
]


# --------------------------------------------------------------------------- #
# metadata header
# --------------------------------------------------------------------------- #

def slurm_job_details():
    """Pull the job's requested TRES and limits from scontrol.

    scontrol is authoritative for what was *asked for* (AllocTRES spells out
    cpu/mem/gres), which the env vars only partly reveal. It stops working once
    the job ends, hence recording it here.
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return {}
    try:
        out = subprocess.check_output(["scontrol", "show", "job", "-o", job_id],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
    except (subprocess.SubprocessError, OSError):
        return {}
    fields = dict(re.findall(r"(\w+)=(\S*)", out))
    return {key: fields[key] for key in
            ("JobName", "Partition", "Account", "QOS", "AllocTRES", "ReqTRES",
             "TimeLimit", "NumCPUs", "NumNodes", "MinMemoryNode", "NodeList")
            if key in fields}


def collect_metadata(alloc):
    """Everything needed to reconstruct the allocation and the node it ran on."""
    meta = {
        "profiler_version": "2",
        "started": datetime.now().isoformat(timespec="seconds"),
        "scope": alloc.scope,
        "hostname": os.uname().nodename,
        "user": psutil.Process().username(),
    }

    # What we were allocated -- the denominator for every percentage above.
    meta["alloc_cpus"] = alloc.n_cpus
    meta["alloc_mem_GB"] = f"{alloc.mem_limit * GB:.2f}"
    if alloc.cgroup:
        meta["alloc_cpu_list"] = (alloc.cgroup / "cpuset.cpus.effective").read_text().strip()
        meta["alloc_cgroup"] = alloc.cgroup
    if alloc.gpus:
        meta["alloc_gpus"] = "; ".join(
            f"{name} ({to_float(mem_total) * 1e-3:.0f} GB)"
            for _, name, mem_total, _ in alloc.gpus)
        meta["alloc_gpu_count"] = len(alloc.gpus)
        meta["gpu_driver_version"] = alloc.gpus[0][3]
    else:
        meta["alloc_gpus"] = "none"

    for key, value in slurm_job_details().items():
        meta[f"slurm_{key}"] = value
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                "SLURM_SUBMIT_DIR", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(var):
            meta[var.lower()] = os.environ[var]

    # Node specs: not our budget, but they explain contention and let a log be
    # compared against one from a different node type.
    cpu_model = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
    meta["node_cpu_model"] = cpu_model.group(1).strip() if cpu_model else "unknown"
    meta["node_cpus_total"] = psutil.cpu_count()
    meta["node_mem_total_GB"] = f"{psutil.virtual_memory().total * GB:.1f}"
    return meta


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Job-scoped resource logger.")
    parser.add_argument("-o", "--output", default="resource_log.csv", help="Output log file")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Sampling interval seconds")
    parser.add_argument("--once", action="store_true", help="Take a single sample and exit")
    parser.add_argument("--print", action="store_true", help="Print output also to console")
    args = parser.parse_args()

    alloc = Allocation()
    sampler = Sampler(alloc)
    header = ",".join(COLUMNS)
    preamble = [f"# {k}: {v}" for k, v in collect_metadata(alloc).items()]

    try:
        with open(args.output, "w", buffering=1) as f:
            for line in (*preamble, header):
                f.write(line + "\n")
                if args.print:
                    print(line)

            while True:
                loop_start = time.monotonic()
                row = ",".join(sampler.sample())
                f.write(row + "\n")
                if args.print:
                    print(row)
                if args.once:
                    break

                sleep_for = args.interval - (time.monotonic() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\nScript stopped manually.")


if __name__ == "__main__":
    main()
