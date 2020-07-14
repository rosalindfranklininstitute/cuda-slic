#!/bin/bash

make

if [ $? -eq 0 ]
then
  echo "The script ran ok"
  exit 1
else
  echo "The script failed" >&2
  exit 0
fi


