# use one of the images from this repository: https://github.com/centreborelli/ipol-docker-images/
FROM registry.ipol.im/ipol:v1-py3.8

# install additional debian packages
# COPY .ipol/packages.txt packages.txt
# RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/* && rm packages.txt

# copy the requirements.txt and install python packages
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt


ENV bin /workdir/bin/
RUN mkdir -p $bin/tmp


WORKDIR $bin

# RUN echo "Hi I'm here"

# copy the code to $bin
COPY . .


# the execution will happen in the folder /workdir/exec
# it will be created by IPOL

# some QoL tweaks
ENV PYTHONDONTWRITEBYTECODE 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH $bin:$PATH

# $HOME is writable by the user `ipol`, but 
ENV HOME /home/ipol
# chmod 777 so that any user can use the HOME, in case the docker is run with -u 1001:1001
RUN groupadd -g 1000 ipol && useradd -m -u 1000 -g 1000 ipol -d $HOME && chmod -R 777 $HOME
USER ipol