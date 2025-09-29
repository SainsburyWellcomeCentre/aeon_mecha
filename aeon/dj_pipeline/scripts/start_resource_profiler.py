""" Simple periodic system resource logger: CPU, memory, network, GPU.

This script logs system resource usage at regular intervals to a CSV file.

Usage:
    python start_resource_profiler.py -o resource_use.csv & PROFILER_PID=$!
    # Run the script to profile here
    kill $PROFILER_PID

Adrian 2025-09-29
"""


import time
import psutil
import argparse
import subprocess
from datetime import datetime

#!/usr/bin/env python3


def get_usage_sample(prev_net=None):
    """Get a sample of system usage statistics.
    
    Args:
        prev_net: Previous net_io_counters object or None.
        interval: Time in seconds since last sample.
    Returns:
        A tuple (row_str, net) where row_str is a comma-separated string of stats,
        and net is the current net_io_counters object to be used in the next call.
     """
    
    # CPU
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_times = psutil.cpu_times_percent(interval=None)
    vm = psutil.virtual_memory()
    
    # Network
    net = psutil.net_io_counters()
    if prev_net is None:
        delta_sent = delta_recv = 0.0
    else:
        delta_sent = net.bytes_sent - prev_net.bytes_sent
        delta_recv = net.bytes_recv - prev_net.bytes_recv
    
    # GPU
    gpu_stats = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip().split(', ')   # returns e.g. ['0', '0', '40536', '25', '48.29']

    # Create comma-separated row
    row = [
        datetime.now().isoformat(),
        f"{cpu_pct:.1f}",
        f"{cpu_times.user:.1f}",
        f"{cpu_times.system:.1f}",
        f"{cpu_times.idle:.1f}",
        f"{vm.percent:.1f}",
        f"{(vm.used * 1e-9):.1f}",
        f"{(vm.total * 1e-9):.1f}",
        f"{(vm.available * 1e-9):.1f}",
        f"{(delta_sent * 1e-6):.1f}",
        f"{(delta_recv * 1e-6):.1f}",
        f"{gpu_stats[0]}",
        f"{(int(gpu_stats[1])*1e-3):.1f}",
        f"{(int(gpu_stats[2])*1e-3):.1f}",
        f"{gpu_stats[3]}",
        f"{(float(gpu_stats[4])):.1f}"
    ]
    row_str = ",".join(row)
    return row_str, net

def get_usage_header():
    """Get the CSV header for the usage statistics."""
    header = [
        "timestamp",
        "cpu_percent",
        "cpu_user_percent",
        "cpu_system_percent",
        "cpu_idle_percent",
        "mem_percent",
        "mem_used_GB",
        "mem_total_GB",
        "mem_available_GB",
        "net_bytes_sent_MB",
        "net_bytes_recv_MB",
        "gpu_utilization_percent",
        "gpu_memory_used_GB",
        "gpu_memory_total_GB",
        "gpu_temperature_C",
        "gpu_power_draw_W",
    ]
    header_str = ",".join(header)
    return header_str


def main():
    parser = argparse.ArgumentParser(description="Periodic system resource logger.")
    parser.add_argument("-o", "--output", default="resource_log.csv", help="Output log file")
    parser.add_argument("-i", "--interval", type=float, default=1.0, help="Sampling interval seconds")
    parser.add_argument("--once", action='store_true', help="Take a single sample and exit")
    parser.add_argument("--print", action='store_true', help="Print output also to console")
    args = parser.parse_args()

    header = get_usage_header()

    try:
        with open(args.output, "w", buffering=1) as f:
            f.write(header + "\n")
            if args.print:
                print(header)
                
            prev_net = None
            while True:
                loop_start = time.time()
                new_row, prev_net = get_usage_sample(prev_net)
                f.write(new_row + "\n")
                if args.print:
                    print(new_row)
                if args.once:
                    break
                    
                elapsed = time.time() - loop_start
                sleep_for = args.interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                    
    except KeyboardInterrupt:
        print("\nScript stopped manually.")

if __name__ == "__main__":
    main()
    