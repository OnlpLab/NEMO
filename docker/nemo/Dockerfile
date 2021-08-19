FROM debian:bullseye

RUN apt-get update
RUN apt-get --no-install-recommends -y install git \
	ca-certificates
RUN update-ca-certificates

RUN mkdir -p /nemo \
	&& cd /nemo \ 
	&& git clone --depth 1 https://github.com/OnlpLab/NEMO.git

WORKDIR /nemo/NEMO/

RUN cd /nemo/NEMO/ \ 
	&& gunzip data/*.gz
	
RUN apt-get --no-install-recommends -y install python3-pip
RUN rm -rf /var/lib/apt/lists/*
RUN cd /nemo/NEMO/ \ 
	&& pip install -r requirements_cpu_only.txt -f https://download.pytorch.org/whl/torch_stable.html
