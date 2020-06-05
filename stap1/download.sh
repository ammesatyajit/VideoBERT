#!/bin/bash

while read id; do
  echo "processing: $id"
  youtube-dl "http://www.youtube.com/watch?v=$id" --output "%(id)s.%(ext)s"
  sleep 2
done < ids.txt
