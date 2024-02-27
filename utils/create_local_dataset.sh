LOCALDIR=$1  # путь сохранения датасета
DATASET_TYPE=$2  # тип датасета, GameEngineData или DownscaleData
LR=$3  # значение низкого разрешения
HR=$4  # значение высокого разрешения
DOWNLOAD=$5  # скачать датасет

set -e

if [[ "$DOWNLOAD" = "true" ]]; then
  mkdir -p "$LOCALDIR/tmp"

  huggingface-cli download epishchik/srgb \
    --local-dir "$LOCALDIR/tmp" \
    --local-dir-use-symlinks True \
    --repo-type dataset \
    --include "data/$DATASET_TYPE/*$LR.tar.gz" "data/$DATASET_TYPE/*$HR.tar.gz"
fi

mkdir -p "$LOCALDIR/$DATASET_TYPE/$LR/train"
mkdir -p "$LOCALDIR/$DATASET_TYPE/$LR/val"
mkdir -p "$LOCALDIR/$DATASET_TYPE/$HR/train"
mkdir -p "$LOCALDIR/$DATASET_TYPE/$HR/val"

mkdir -p "$LOCALDIR/tmp_files"

for DIR in $LOCALDIR/tmp/data/$DATASET_TYPE/*
do
  # train low res archive
  tar -xzf "$DIR/train-$LR.tar.gz" -C "$LOCALDIR/tmp_files"
  for FILE in $LOCALDIR/tmp_files/*
  do
    mv "$FILE" "$LOCALDIR/$DATASET_TYPE/$LR/train/${DIR##*/}_${FILE##*/}"
  done
  rm "$DIR/train-$LR.tar.gz"

  # train high res archive
  tar -xzf "$DIR/train-$HR.tar.gz" -C "$LOCALDIR/tmp_files"
  for FILE in $LOCALDIR/tmp_files/*
  do
    mv "$FILE" "$LOCALDIR/$DATASET_TYPE/$HR/train/${DIR##*/}_${FILE##*/}"
  done
  rm "$DIR/train-$HR.tar.gz"

  # val low res archive
  tar -xzf "$DIR/val-$LR.tar.gz" -C "$LOCALDIR/tmp_files"
  for FILE in $LOCALDIR/tmp_files/*
  do
    mv "$FILE" "$LOCALDIR/$DATASET_TYPE/$LR/val/${DIR##*/}_${FILE##*/}"
  done
  rm "$DIR/val-$LR.tar.gz"

  # val high res archive
  tar -xzf "$DIR/val-$HR.tar.gz" -C "$LOCALDIR/tmp_files"
  for FILE in $LOCALDIR/tmp_files/*
  do
    mv "$FILE" "$LOCALDIR/$DATASET_TYPE/$HR/val/${DIR##*/}_${FILE##*/}"
  done
  rm "$DIR/val-$HR.tar.gz"

  # check total number of files in order (train lr, train hr, val lr, val hr)
  echo ${DIR##*/}
  ls "$LOCALDIR/$DATASET_TYPE/$LR/train" | wc -l
  ls "$LOCALDIR/$DATASET_TYPE/$HR/train" | wc -l
  ls "$LOCALDIR/$DATASET_TYPE/$LR/val" | wc -l
  ls "$LOCALDIR/$DATASET_TYPE/$HR/val" | wc -l
done

rm -rf "$LOCALDIR/tmp"
rm -rf "$LOCALDIR/tmp_files"
