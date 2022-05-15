#!/bin/bash

count=$(ls emo-db/*.wav | wc -l)
echo "Will be converted $count files"
sleep 1

i=0
for file in emo-db/*.wav; do
  i=$((i+1))
  echo -e "$i / $count:\t${file}"
  
  # covert to wav with 1 audio channel and sampling freqency 11 kHz 
  ffmpeg -hide_banner -loglevel warning -y -i "$file" -ac 1 -ar 11025 "emo-db/c_${file##*/}"
  
  rm "$file"
  mv "emo-db/c_${file##*/}" "$file"
done
