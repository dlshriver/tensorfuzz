#!/usr/bin/env bash

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
BASEDIR=$(realpath $DIR/../../)

source $BASEDIR/.venv/bin/activate
export PYTHONPATH=$BASEDIR:$PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=3

python $DIR/localrobustness_fuzzer.py $@
