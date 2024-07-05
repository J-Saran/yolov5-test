FROM tiangolo/uvicorn-gunicorn:python3.9-slim
RUN mkdir /yolov5-fastapi-test
COPY ./requirements.txt /yolov5-fastapi-test/requirements.txt
RUN apt-get update
RUN apt-get install gcc ffmpeg libsm6 libxext6 openssh-client -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /yolov5-fastapi-test/requirements.txt
COPY . /yolov5-fastapi-test
WORKDIR /yolov5-fastapi-test
CMD uvicorn main:app  --host=0.0.0.0 --port=${PORT:-8000}
