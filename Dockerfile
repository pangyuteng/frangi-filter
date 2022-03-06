FROM tensorflow/tensorflow:2.6.1-gpu-jupyter

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install tensorflow_addons
