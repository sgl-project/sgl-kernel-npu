#!/bin/bash
sudo chmod a+rw /var/run/docker.sock

CONTAINER_NAME="sglang_kernel_ci_a3"

if docker ps -a --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists. Removing it..."

    if docker ps --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
        echo "Stopping container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME"
    fi

    docker rm "$CONTAINER_NAME"
    echo "Container '$CONTAINER_NAME' has been removed."
fi
