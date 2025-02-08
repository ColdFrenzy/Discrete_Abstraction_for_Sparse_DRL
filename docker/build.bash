#!/bin/bash

UPAR="--build-arg UID=`id -u` --build-arg GID=`id -g`"

docker build $UPAR -t marl -f Dockerfile .

