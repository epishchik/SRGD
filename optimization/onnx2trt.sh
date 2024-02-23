#!/bin/bash

TEMP=$(getopt -o o:s:m:v:h --long onnx:,save:,model:,version:,help,optr::,maxr:: -n 'onnx2trt' -- "$@")
if [[ $? != 0 ]] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"

USAGE=$(cat <<-EOM
USAGE: ./onnx2trt.sh <-o path> <-s path> <-m name> [--optr=h,w] [--maxr=h,w] [-h]
PARAMETERS:
  -o|--onnx - path to folder with onnx model.
  -s|--save - path to folder where to save tensorrt engine.
  -m|--model - model filename without extenstion.
  -v|--version - model version.
  --optr - optimal resolution in format --optr=h,w or in format --optr=r, where r=h=w.
  --maxr - maximum resolution in format --maxr=h,w or in format --maxr=r, where r=h=w.
  -h|--help - print help message.
EOM
)

ONNX=-1
SAVE=-1
MODEL=-1
VERSION=1
H_OPT=270
H_MAX=540
W_OPT=480
W_MAX=960

while true; do
  case $1 in
    -o | --onnx ) ONNX=$2; shift 2 ;;
    -s | --save ) SAVE=$2; shift 2 ;;
    -m | --model ) MODEL=$2; shift 2 ;;
    -v | --version ) VERSION=$2; shift 2 ;;
    -h | --help ) echo -e "$USAGE"; exit 0 ;;
    --optr ) H_OPT=${2%,*}; W_OPT=${2#*,}; shift 2 ;;
    --maxr ) H_MAX=${2%,*}; W_MAX=${2#*,}; shift 2 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [[ $ONNX == -1 ]] ; then
  echo "ERROR: -o|--onnx parameter required."
  echo "$USAGE"
  exit -1
fi

if [[ $SAVE == -1 ]] ; then
  echo "ERROR: -s|--save parameter required."
  echo "$USAGE"
  exit -1
fi

if [[ $MODEL == -1 ]] ; then
  echo "ERROR: -m|--model parameter required."
  echo "$USAGE"
  exit -1
fi

if [[ -z $H_OPT ]] ; then
  echo "ERROR: specified --optr parameter is incorrect."
  echo "$USAGE"
  exit -1
fi

if [[ -z $W_OPT ]] ; then
  echo "ERROR: specified --optr parameter is incorrect."
  echo "$USAGE"
  exit -1
fi

if [[ -z $H_MAX ]] ; then
  echo "ERROR: specified --maxr parameter is incorrect."
  echo "$USAGE"
  exit -1
fi

if [[ -z $W_MAX ]] ; then
  echo "ERROR: specified --maxr parameter is incorrect."
  echo "$USAGE"
  exit -1
fi

if [[ "${ONNX: -1}" == "/" ]] ; then
  ONNX=${ONNX%?}
fi

if [[ "${SAVE: -1}" == "/" ]] ; then
  SAVE=${SAVE%?}
fi

if [[ ! -f $ONNX/$MODEL.onnx ]] ; then
  echo "ERROR: can't find $ONNX/$MODEL.onnx file."
  echo "$USAGE"
  exit -1
fi

if [[ ! -d $SAVE/$MODEL/$VERSION ]] ; then
  mkdir -p $SAVE/$MODEL/$VERSION
fi

# you can use the --fp8 option if your GPU supports it
trtexec --onnx=$ONNX/$MODEL.onnx \
  --saveEngine=$SAVE/$MODEL/$VERSION/model.plan \
  --minShapes=lr:1x3x1x1 \
  --optShapes=lr:1x3x"$H_OPT"x"$W_OPT" \
  --maxShapes=lr:1x3x"$H_MAX"x"$W_MAX" \
  --noTF32 \
  --fp16 \
  --int8 \
  --useCudaGraph \
  --verbose
