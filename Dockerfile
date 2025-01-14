FROM --platform=linux/amd64 python:3.7-slim-bookworm

COPY requirements.txt /requirements.txt

RUN pip install --require-hashes --no-deps -r /requirements.txt

ENV PYTHONPATH=/usr/local/bin/python

CMD ["python"]