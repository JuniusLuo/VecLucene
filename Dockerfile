
# get the required python packages
FROM python:3.11-slim as requirements-stage

WORKDIR /tmp

RUN pip install poetry

COPY ./pyproject.toml ./poetry.lock /tmp/

RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


# build
FROM python:3.11-slim

# 1. build pylucene
# building on mac, default jdk does not work for JCC on aarch64/arm64,
# pylucene-9.4.1/jcc/setup.py line 197, LFLAGS does not have linux/aarch64.
#RUN apt-get update && apt-get install -y default-jdk

# https://adoptium.net/blog/2021/12/eclipse-temurin-linux-installers-available/
RUN apt-get update && apt-get install -y wget apt-transport-https gnupg
RUN wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | apt-key add -
RUN echo "deb https://packages.adoptium.net/artifactory/deb \
	$(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" \
	| tee /etc/apt/sources.list.d/adoptium.list
RUN apt-get update && apt-get install -y temurin-17-jdk

RUN apt-get install -y build-essential

# download and build pylucene
WORKDIR /code/pylucene
RUN wget -O - https://downloads.apache.org/lucene/pylucene/pylucene-9.4.1-src.tar.gz \
	| tar -xz --strip-components=1
RUN cd jcc \
    && JCC_JDK=/usr/lib/jvm/$(ls /usr/lib/jvm) python setup.py build install
RUN make all install JCC='python -m jcc --shared' PYTHON=python NUM_FILES=16
#RUN make all test install JCC='python -m jcc --shared' PYTHON=python NUM_FILES=16

WORKDIR /code
RUN rm -rf pylucene


# 2. install VecLucene python packages
WORKDIR /code/VecLucene
COPY . /code/VecLucene/

COPY --from=requirements-stage /tmp/requirements.txt /code/VecLucene/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/VecLucene/requirements.txt

EXPOSE 8080

CMD ["sh", "-c", "uvicorn server.server:app --host 0.0.0.0 --port 8080"]
