#!/bin/bash

if [[ -d "$1" ]] && [[ -d "$2" ]]; then
    for i in $1*.wav; do
        ffmpeg -y -i "$i" -c:a libgsm -ar 8000 -ab 13000 -ac 1 -f gsm temp.gsm
        ffmpeg -y -i temp.gsm -acodec pcm_s16le -ac 1 -ar 8000 "$2$(basename $i)"
        rm temp.gsm
    done
else
    echo "Please provide an input and output directory"
fi
