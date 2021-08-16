#!/bin/bash

# change to location of `docker-compose.yml` before running.

THIS_FILE=$(basename "$0")
ARG1="$1"

if [[ $ARG1 = "--help" ]] || [[ $ARG1 = "-h" ]]; then
    echo -e "\nNOTE: Change to location of 'docker-compose.yml' before running\n"
    echo "usage: ./$THIS_FILE \"~/.ssh/my_deploy_key\""
    echo "  if ./$THIS_FILE has no input argument, the ssh key is pulled from ~/.ssh/github"
    echo -e "  to docker-compose down: ./$THIS_FILE -d\n"
    exit
fi

if [[ $ARG1 = "-d" ]]; then
    docker-compose down --volumes --remove-orphans
    exit
fi

if [[ -z $ARG1 ]]; then
    SSH_KEY="$(cat ~/.ssh/github)"
    echo "Using private key from ~/.ssh/github"
else
    SSH_KEY="$(cat $ARG1)"
    echo "Using private key from $ARG1"
fi


docker-compose build --build-arg SSH_KEY="${SSH_KEY}"
docker-compose up -d
