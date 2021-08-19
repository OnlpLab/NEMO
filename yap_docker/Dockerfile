FROM golang:1.17-bullseye

RUN apt-get update
RUN apt-get --no-install-recommends -y install \
	bzip2 \
	git

RUN mkdir -p /yap/src \
	&& cd /yap/src \ 
	&& git clone --depth 1 https://github.com/OnlpLab/yap.git

WORKDIR /yap/src/yap/

RUN cd /yap/src/yap \
	&& bunzip2 data/*.bz2
ENV GOPATH=/yap
ENV GO111MODULE=off
RUN cd /yap/src/yap \
	&& go get .
RUN cd /yap/src/yap \
	&& go build .
