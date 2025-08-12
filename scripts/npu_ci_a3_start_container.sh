#!/bin/bash
sudo chmod a+rw /var/run/docker.sock
IMAGE_NAME="swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/sglang:main"
sudo docker pull $IMAGE_NAME

CONTAINER_NAME="sglang_ci_a3"

if docker ps -a --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
    echo "Container '$CONTAINER_NAME' exists. Removing it..."

    if docker ps --format '{{.Names}}' | grep -qw "^${CONTAINER_NAME}$"; then
        echo "Stopping container '$CONTAINER_NAME'..."
        docker stop "$CONTAINER_NAME"
    fi

    docker rm "$CONTAINER_NAME"
    echo "Container '$CONTAINER_NAME' has been removed."
fi

echo "starting sglang npu-A3 container"
docker run -itd \
  --name "$CONTAINER_NAME" \
  --shm-size=20g \
  --net=host \
  --ipc=host \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /home/runner:/home/runner \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  --device=/dev/davinci0:/dev/davinci0 \
  --device=/dev/davinci1:/dev/davinci1 \
  --device=/dev/davinci2:/dev/davinci2 \
  --device=/dev/davinci3:/dev/davinci3 \
  --device=/dev/davinci4:/dev/davinci4 \
  --device=/dev/davinci5:/dev/davinci5 \
  --device=/dev/davinci6:/dev/davinci6 \
  --device=/dev/davinci7:/dev/davinci7 \
  --device=/dev/davinci8:/dev/davinci8 \
  --device=/dev/davinci9:/dev/davinci9 \
  --device=/dev/davinci10:/dev/davinci10 \
  --device=/dev/davinci11:/dev/davinci11 \
  --device=/dev/davinci12:/dev/davinci12 \
  --device=/dev/davinci13:/dev/davinci13 \
  --device=/dev/davinci14:/dev/davinci14 \
  --device=/dev/davinci15:/dev/davinci15 \
  --device=/dev/davinci_manager:/dev/davinci_manager \
  --device=/dev/devmm_svm:/dev/devmm_svm \
  --device=/dev/hisi_hdc:/dev/hisi_hdc \
  "$IMAGE_NAME"

# Check if container start successfully
if [ $? -eq 0 ]; then
  echo "Container $CONTAINER_NAME start successfully"
else
  echo "Container $CONTAINER_NAME start failed, please check if the images exist or permission"
  exit 1
fi
