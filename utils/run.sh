#!/bin/bash

source_init() {
    if [ -s init.sh ]; then
        echo Sourcing: $(realpath init.sh)
        source init.sh
    fi
}

source_init
pushd $1
source_init
echo Running: "${@:2}"
"${@:2}"
