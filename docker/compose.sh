#!/bin/bash

THIS_FILE=$(basename "$0")
ARG1="$1"
DEFAULT_KEY=~/.ssh/aeon_mecha
USAGE="
NOTE: Change to location of 'docker-compose.yml' before running
Usage: ./$THIS_FILE \"~/.ssh/my_deploy_key\"
- If ./$THIS_FILE has no input argument, the ssh key is pulled from $DEFAULT_KEY
- To docker-compose down: ./$THIS_FILE -d

"

if [[ $ARG1 = "--help" ]] || [[ $ARG1 = "-h" ]]; then
    echo "$USAGE"
    exit
fi

if [[ $ARG1 = "-d" ]]; then
    docker-compose down --volumes --remove-orphans
    exit
fi

if [[ -z $ARG1 ]]; then
    SSH_KEY="$(cat $DEFAULT_KEY)"
    echo "Using private key from $DEFAULT_KEY"
else
    SSH_KEY="$(cat $ARG1)"
    echo "Using private key from $ARG1"
fi

# echo $SSH_KEY

if [[ ! -f "docker-compose.yml" ]]; then
    echo "$USAGE"
    echo "ERROR! docker-compose.yml file not found in current location"
    exit 1
fi

docker-compose build --build-arg SSH_KEY="${SSH_KEY}"
docker-compose up -d
