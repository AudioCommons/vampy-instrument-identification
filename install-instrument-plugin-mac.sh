#!/bin/bash

TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

install_requirements() {
    # Install sonic-annotator
    curl -O https://code.soundsoftware.ac.uk/attachments/download/1702/sonic-annotator-1.3-osx.tar.gz
    tar xzf sonic-annotator-1.3-osx.tar.gz
    mkdir -p ~/bin
    mv sonic-annotator/sonic-annotator ~/bin/
    echo 'export PATH=$HOME/bin:$PATH' >> ~/.profile

    # Install VamPy
    curl --location -O http://code.soundsoftware.ac.uk/attachments/download/674/vampy-2.0-osx-108.tar.bz2
    tar xjf vampy-2.0-osx-108.tar.bz2
    mkdir -p ~/Library/Audio/Plug-Ins/Vamp/
    mv vampy-2.0-osx-108/vampy.dylib ~/Library/Audio/Plug-Ins/Vamp/

    # Install Keras
    pip install -U --user six
    echo 'export PYTHONPATH=$HOME/Library/Python/2.7/lib/python/site-packages' >> ~/.profile
    pip install --user Keras h5py

    echo "Installation finished. Close this shell such that the new environment variables can be loaded."
}

install_plugin() {
    curl -O http://www.eecs.qmul.ac.uk/~johan/VampInstrumentIdentification.zip
    unzip VampInstrumentIdentification.zip
    cp -R -f VampInstrumentIdentification/deepdeploy ~/Library/Audio/Plug-Ins/Vamp/
    cp -f VampInstrumentIdentification/VampyInstrumentIdentification.py ~/Library/Audio/Plug-Ins/Vamp/
    cp -f VampInstrumentIdentification/model_th.json ~/Library/Audio/Plug-Ins/Vamp/
    cp -f VampInstrumentIdentification/weights_theano_th.hdf5 ~/Library/Audio/Plug-Ins/Vamp/
    cp -f VampInstrumentIdentification/*.n3 ~
}

case "$1" in
    requirements)
        install_requirements ;;
    plugin)
        install_plugin ;;
    *)
        install_requirements
        install_plugin ;;
esac

cd -
rm -rf $TEMP_DIR
