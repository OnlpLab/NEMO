FROM debian:bullseye

RUN apt-get update
RUN apt-get --no-install-recommends -y install git \
	ca-certificates
RUN update-ca-certificates
	
RUN apt-get --no-install-recommends -y install python3-pip
RUN rm -rf /var/lib/apt/lists/*

COPY . /NEMO

WORKDIR /NEMO/

RUN cd /NEMO/ \ 
	&& gunzip data/*.gz || true
	
RUN cd /NEMO/ \ 
	&& pip install -r requirements_cpu_only.txt -f https://download.pytorch.org/whl/torch_stable.html
