#!/usr/bin/env bash

if [ "$EUID" -ne 0 ]
  then echo "Please run with sudo"
  exit
fi

docker-compose down
rm -rf ./pg_data/

docker-compose build
docker-compose up -d
