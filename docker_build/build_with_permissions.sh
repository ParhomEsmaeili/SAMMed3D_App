#!/bin/bash
docker_tag=testing:sammed3dv1
#docker push ${docker_tag}
docker build --no-cache . -f Dockerfile_With_Permissions \
 -t ${docker_tag} \
 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER=${USER} --network=host
#docker push ${docker_tag}
