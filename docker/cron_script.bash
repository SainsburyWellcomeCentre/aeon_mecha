#!/bin/bash

# This script will be run every 4 hours in a cron job.
# Open up a crontab ('crontab -e') and add the following:
# 0 */4 * * * /path/to/cron_script.bash
# For debugging, run ./cron_script.bash -v
# Create a log file whenever the job gets run.
ROOT_LOG_DIR="/ceph/aeon/aeon/dj_store/logs"
mkdir -p "${ROOT_LOG_DIR}"
LOG_FILE="${ROOT_LOG_DIR}/cron_script_$(date '+%Y%m%d_%H%M%S').log"

verbose=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
    -v | --verbose) verbose=1 ;;
    esac
    shift
done

# Verbose option for debugging
print_verbose() {
    if [ "$verbose" -eq 1 ]; then
        echo "[DEBUG] $1 - $(date '+%Y-%m-%d %H:%M:%S')"
    fi
    echo "[DEBUG] $1 - $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
}

print_verbose "Starting Ingestion..."
cd /nfs/nhome/live/aeon_db/aeon_mecha/docker/

print_verbose "Terminate running workers..."
/usr/local/bin/docker-compose down >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    print_verbose "Workers terminated successfully."
else
    print_verbose "Failed to terminate workers."
fi

if /usr/bin/docker image inspect ghcr.io/sainsburywellcomecentre/aeon_mecha >/dev/null 2>&1; then
    print_verbose "Removing existing aeon_mecha image..."
    /usr/bin/docker image rm ghcr.io/sainsburywellcomecentre/aeon_mecha >> "$LOG_FILE" 2>&1
fi

print_verbose "Restart workers..."
/usr/local/bin/docker-compose up --detach >> "$LOG_FILE" 2>&1
if [ $? -eq 0 ]; then
    print_verbose "Workers restarted successfully."
else
    print_verbose "Failed to restart workers."
fi

# Clean up old logs (older than 30 days)
print_verbose "Cleaning up old logs..."
find "${ROOT_LOG_DIR}" -mtime +30 -delete;
