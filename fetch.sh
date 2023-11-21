#!/bin/bash

wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzf train-clean-100.tar.gz
tar -xzf test-clean.tar.gz
rm *.tar.gz