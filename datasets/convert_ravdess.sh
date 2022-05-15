#!/bin/bash

# ./convert_ravdess.sh

count=$(ls ravdess/*.wav | wc -l)
echo "Will be converted $count files"
sleep 1

i=0
for file in ravdess/*.wav; do
  i=$((i+1))
  echo -e "$i / $count:\t${file}"
  
  duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${file}")
  cut=$(echo "x=$duration-0.8; if(x<1) print 0; x" | bc)
  ffmpeg -hide_banner -loglevel warning -y -i "$file" -ac 1 -ar 11025 -ss 00:00:00.900 -to "$cut" "ravdess/c_${file##*/}"
  
  rm "$file"
  mv "ravdess/c_${file##*/}" "$file"
done
