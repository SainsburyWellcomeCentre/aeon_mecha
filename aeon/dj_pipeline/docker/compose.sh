#!/bin/bash
# Joseph Burling joseph@datajoint.com

# get this script file (no symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
THIS_FILE=$(basename "$0")

# defaults
DEPLOY_KEY="~/.ssh/aeon_mecha"
N_WORKERS_L=0
N_WORKERS_M=0
N_WORKERS_H=0
DOWN_COMPOSE=0
UP_COMPOSE=0
COMPOSE_FILE="docker-compose.yml"
N_ARGS=$#

# program help documentation
USAGE="
$THIS_FILE : start/stop containerized aeon ingestion routine

Usage: $THIS_FILE [options]

options:
    -h, --help              show usage help
    -d, --down              docker compose down, removing orphans
    -k, --key=DEPLOY_KEY    specify path to private deploy key (default=$DEPLOY_KEY)
        --low=N_WORKERS_L   number of workers for low priority tasks
        --mid=N_WORKERS_M   number of workers for mid priority tasks
        --high=N_WORKERS_H  number of workers for high priority tasks
"

# handle input arguments
while test $# -gt 0; do
    case "$1" in
    -h | --help)
        echo "$USAGE"
        exit 0
        ;;
    -d | --down)
        export DOWN_COMPOSE=1
        shift
        ;;
    -k)
        shift
        if test $# -gt 0; then
            export DEPLOY_KEY=$1
        else
            echo "no deploy key file specified"
            exit 1
        fi
        shift
        ;;
    --key*)
        export DEPLOY_KEY=$(echo $1 | sed -e 's/^[^=]*=//g') # sed 's/[^=]*//'
        if [ $DEPLOY_KEY == "--key" ]; then
            shift
            export DEPLOY_KEY=$1
        fi
        shift
        ;;
    --low*)
        export N_WORKERS_L=$(echo $1 | sed -e 's/^[^=]*=//g')
        if [ $N_WORKERS_L == "--low" ]; then
            shift
            export N_WORKERS_L=$1
        fi
        shift
        ;;
    --mid*)
        export N_WORKERS_M=$(echo $1 | sed -e 's/^[^=]*=//g')
        if [ $N_WORKERS_M == "--mid" ]; then
            shift
            export N_WORKERS_M=$1
        fi
        shift
        ;;
    --high*)
        export N_WORKERS_H=$(echo $1 | sed -e 's/^[^=]*=//g')
        if [ $N_WORKERS_H == "--high" ]; then
            shift
            export N_WORKERS_H=$1
        fi
        shift
        ;;
    *)
        echo "invalid option: $1"
        exit 1 # break
        ;;
    esac
done

eval DEPLOY_KEY=$DEPLOY_KEY

# check that target file exists
if [[ ! -f $COMPOSE_FILE ]]; then
    if [[ -f $SCRIPT_DIR/$COMPOSE_FILE ]]; then
        cd "$SCRIPT_DIR"
        echo "Changing to location of $COMPOSE_FILE found at: $SCRIPT_DIR"
    else
        echo "$USAGE"
        echo "ERROR! $COMPOSE_FILE file not found in current location: $SCRIPT_DIR"
        exit 1
    fi
fi

# optionally compose down before building (or just compose down)
if [[ $DOWN_COMPOSE -eq 1 ]]; then
    echo -e "Running command:\n  docker-compose down --remove-orphans"
    docker-compose down --remove-orphans

    # only decompose then exit if no other args besides -d or --down
    if [[ $N_ARGS -eq 1 ]]; then
        exit
    fi
fi

# no workers to run so do rebuild and exit
if [[ $N_WORKERS_L -eq 0 ]] && [[ $N_WORKERS_M -eq 0 ]] && [[ $N_WORKERS_H -eq 0 ]]; then
    echo -e "Running command:\n  docker-compose build --build-arg SSH_KEY=\"\$(cat $DEPLOY_KEY)\""
    docker-compose build --build-arg SSH_KEY="$(cat $DEPLOY_KEY)"
    exit
fi

echo "Running command:
  SSH_KEY=\"\$(cat $DEPLOY_KEY)\" docker-compose up --build -d \\
    --scale aeon_high=$N_WORKERS_H \\
    --scale aeon_mid=$N_WORKERS_M \\
    --scale aeon_low=$N_WORKERS_L \\"

SSH_KEY="$(cat $DEPLOY_KEY)" docker-compose up --build -d \
    --scale aeon_high=$N_WORKERS_H \
    --scale aeon_mid=$N_WORKERS_M \
    --scale aeon_low=$N_WORKERS_L
