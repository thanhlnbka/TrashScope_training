FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04


RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0

RUN pip install --upgrade pip
RUN pip install ultralytics
RUN pip install torch torchvision torchaudio tensorboard matplotlib tensorboard

RUN apt-get update && apt-get install -y libx11-dev libxcb-xinerama0-dev libqt5gui5
RUN apt-get update && apt-get install -y zip vim

WORKDIR /workspace

COPY ./obj_cls_other/requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

CMD ["bash"]