#!/bin/bash

while read id; do
  echo "processing: $id"
  youtube-dl -f --match-filter 'duration < 900' "http://www.youtube.com/watch?v=$id" --add-header "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36" --output "VID_%(id)s.%(ext)s"
  sleep 2
done < ids.txt
