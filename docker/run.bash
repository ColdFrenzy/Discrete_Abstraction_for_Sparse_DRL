#!/bin/bash

if [ "$1" == "tf" ]; then
  IMAGE="not_valid"
else
  IMAGE="dadrl"
fi

docker run -it -d \
    -u $(id -u):$(id -g)  \
    --name dadrl --rm \
    --privileged \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v `pwd`/..:/opt/dadrl \
    -w /opt/dadrl \
    $IMAGE

sleep 5

docker exec -it marl bash -ci "cd /opt/dadrl && pip install -e ."

docker exec -it marl tmux a


