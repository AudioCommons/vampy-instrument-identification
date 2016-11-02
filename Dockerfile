FROM gw000/keras:1.1.1-py2-th-cpu
MAINTAINER Johan Pauwels
RUN apt-get -y update && apt-get install -y \
    libfftw3-double3 \
    libfishsound1 \
    libid3tag0 \
    liblrdf0 \
    libmad0 \
    liboggz2 \
    libqt5network5 \
    libqt5xml5 \
    libsamplerate0 \
    libsndfile1 \
    libsord-0-0 \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

# Install sonic-annotator 1.2
WORKDIR /usr/local/bin
ADD https://code.soundsoftware.ac.uk/attachments/download/1619/sonic-annotator-linux64-1.2.tar.bz2 .
RUN tar xjf sonic-annotator-linux64-1.2.tar.bz2 && rm sonic-annotator-linux64-1.2.tar.bz2 \
&&  echo PATH=$PATH:$(ls -d $PWD/sonic-annotator) >> ~/.bashrc

# Compile VamPy from source
ADD https://code.soundsoftware.ac.uk/attachments/download/1520/vamp-plugin-sdk-2.6.tar.gz .
RUN tar xzf vamp-plugin-sdk-2.6.tar.gz && rm vamp-plugin-sdk-2.6.tar.gz
WORKDIR vamp-plugin-sdk-2.6
RUN ./configure --disable-programs && make sdkstatic
WORKDIR ..
ADD http://code.soundsoftware.ac.uk/attachments/download/672/vampy-2.0.tar.bz2 .
RUN tar xjf vampy-2.0.tar.bz2 && rm vampy-2.0.tar.bz2
WORKDIR vampy-2.0
RUN sed -i 's/python2.6/python2.7/g' Makefile.linux && \
    sed -i 's/-fPIC/-fPIC -I..\/vamp-plugin-sdk-2.6/g' Makefile.linux && \
    sed -i 's/plugin.map/plugin.map -L..\/vamp-plugin-sdk-2.6/g' Makefile.linux
RUN make -f Makefile.linux && make -f Makefile.linux install
WORKDIR ..
RUN rm -rf vampy-2.0 vamp-plugin-sdk-2.6 

# Copy VampyInstrumentIdentification
WORKDIR /usr/local/lib/vamp
COPY model_th.json ./
COPY weights_theano_th.hdf5 ./
COPY VampyInstrumentIdentification.py ./
COPY *.n3 /root/
COPY deepdeploy deepdeploy

WORKDIR /srv