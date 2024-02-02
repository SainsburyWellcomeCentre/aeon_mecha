#!/bin/bash

# This script will be run every 4 hours in a cron job.
# Open up a crontab ('crontab -e') and add the following:
# 0 */4 * * * /path/to/cron_script.bash 
# For debugging, run ./cron_script.bash -v

verbose=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -v|--verbose) verbose=1 ;;
    esac
    shift
done

# Verbose option for debugging
print_verbose() {
    if [ "$verbose" -eq 1 ]; then
        printf "[DEBUG] %s - %s\n\n" "$1" "$(date '+%Y-%m-%d %H:%M:%S')"
    fi
}

print_verbose "Starting Ingestion..."
cd /nfs/nhome/live/aeon_db/aeon_mecha/docker/

if docker image inspect ghcr.io/sainsburywellcomecentre/aeon_mecha > /dev/null 2>&1; then
    print_verbose "Removing existing aeon_mecha image..."
    docker image rm ghcr.io/sainsburywellcomecentre/aeon_mecha
fi

print_verbose "Terminate worker containers..."
docker-compose down
if [ $? -eq 0 ]; then
    print_verbose "Workers terminated successfully."
else
    print_verbose "Failed to terminate workers."
fi

print_verbose "Restart containers..."
docker-compose up --detach
if [ $? -eq 0 ]; then
    print_verbose "Containers restarted successfully."
else
    print_verbose "Failed to restart containers."
fi

