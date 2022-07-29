FROM nvcr.io/nvidia/pytorch:21.12-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install transformers
RUN pip3 install opencv-python 
RUN pip3 install scipy
RUN pip3 install pillow
RUN pip3 install tensorboard
RUN pip3 install torchmetrics
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update
RUN apt-get install libgl1
RUN pip3 install fastseg
RUN pip3 install tqdm

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user