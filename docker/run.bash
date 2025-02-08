#!/bin/bash

if [ "$1" == "tf" ]; then
  IMAGE="not_valid"
else
  IMAGE="marl"
fi

docker run -it -d \
    -u $(id -u):$(id -g)  \
    --name marl --rm \
    --privileged \
    --net=host \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $HOME/.Xauthority:/root/.Xauthority:rw \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -v `pwd`/..:/opt/multi-agent-rl-rm \
    -v $HOME/src/RL/gym-sapientino:/opt/gym-sapientino \
    -w /opt/multi-agent-rl-rm \
    $IMAGE

sleep 5

docker exec -it marl bash -ci "cd /opt/multi-agent-rl-rm && pip install -e ."

docker exec -it marl tmux a


