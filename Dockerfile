FROM continuumio/miniconda3
MAINTAINER Ben Zimmer <ben.d.zimmer@gmail.com>

COPY handwriting36-linux.yml /home/shared/
RUN conda env create --file /home/shared/handwriting36-linux.yml
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender1
