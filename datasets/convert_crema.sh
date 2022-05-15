#!/bin/bash

count=$(ls crema-d/*.wav | wc -l)
echo "Will be converted $count files"
sleep 1

i=0
for file in crema-d/*.wav; do
  i=$((i+1))
  echo -e "$i / $count:\t${file}"
  
  # covert to wav with 1 audio channel and sampling freqency 11 kHz 
  ffmpeg -hide_banner -loglevel warning -y -i "$file" -ac 1 -ar 11025 "crema-d/c_${file##*/}"
  
  rm "$file"
  mv "crema-d/c_${file##*/}" "$file"
done
