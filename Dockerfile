FROM python:3.12-slim

WORKDIR /workdir
COPY src/model.py src/preprocessing.py src/__init__.py requirements.txt  ./
RUN pip install -r requirements.txt