FROM debian:latest

RUN apt-get -y update && apt-get install -y git python3-pip python3-dev python3-tk vim procps curl

ADD . /face_classifier

WORKDIR face_classifier

RUN pip3 install -r requirements.txt
ENV PYTHONPATH=$PYTHONPATH:src
ENV FACE_CLASSIFIER_PORT=8084
EXPOSE $FACE_CLASSIFIER_PORT

ENTRYPOINT ["python3"]
CMD ["faces.py"]
