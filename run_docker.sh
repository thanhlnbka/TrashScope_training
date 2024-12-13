docker run --name trashscope-training \
  --runtime=nvidia \
  --network host \
  --gpus all -it --restart always \
  -v $(pwd):/workspace \
  --memory="11g" --cpus="8" \
  --ipc=host \
  -e DISPLAY=$DISPLAY -e XAUTHORITY=$XAUTHORITY -v $XAUTHORITY:$XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 \
  ultralytics-docker
