#!/bin/bash
SYSTEM_TYPE=$(uname)
user=`id -n -u`
GID=`id -g`

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
if [ -e "$SCRIPT_DIR/.hub_token" ]; then
    TOKEN=`cat "$SCRIPT_DIR/.hub_token"`
else
    TOKEN=""
fi
BUILD="latest"
for argwhole in "$@"; do
    IFS='=' read -r -a array <<< "$argwhole"
    arg="${array[0]}"
    val=$(printf "=%s" "${array[@]:1}")
    val=${val:1}
    case "$arg" in
        --hub_token) TOKEN="$val";;
        --build) BUILD="$val"
    esac
done

if [ "$TOKEN" != "" ]; then
    if [ "$TOKEN" != "public" ]; then
        echo "$TOKEN" | docker login -u jferlez --password-stdin
        if [ $? != 0 ]; then
            echo "ERROR: Unable to login to DockerHub using available access token! Quitting..."
            exit 1
        fi
    fi
    docker pull jferlez/fastbatllnn-deps:$BUILD
    if [ $? != 0 ]; then
        echo "ERROR: docker pull command failed! Quitting..."
        exit 1
    fi
    PROCESSING="s/fastbatllnn-deps:local/jferlez\/fastbatllnn-deps:$BUILD/"
    cd "$SCRIPT_DIR"
    echo "$TOKEN" > .hub_token
else
    cd "$SCRIPT_DIR/DockerDeps"
    docker build -t fastbatllnn-deps:local .
    PROCESSING="s/fastbatllnn/fastbatllnn/"
fi


cd "$SCRIPT_DIR"

if [ $SYSTEM_TYPE = "Darwin" ];
then
    CORES=$(( `sysctl -n hw.ncpu` / 2 ))
    PYTHON="python3.10"
else
    CORES_PER_SOCKET=`lscpu | grep "Core(s) per socket:" | sed -e 's/[^0-9]//g'`
    SOCKETS=`lscpu | grep "Socket(s):" | sed -e 's/[^0-9]//g'`
    CORES=$(( $CORES_PER_SOCKET * $SOCKETS ))
    PYTHON=""
fi

cat Dockerfile | sed -u -e $PROCESSING | docker build --no-cache --build-arg USER_NAME=$user --build-arg UID=$UID --build-arg GID=$GID --build-arg CORES=$CORES -t fastbatllnn-run:${user} -f- .
