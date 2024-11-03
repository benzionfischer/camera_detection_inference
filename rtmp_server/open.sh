#!/bin/bash

docker pull tiangolo/nginx-rtmp
docker run -d --name rtmp-server -p 1935:1935 -p 80:80 tiangolo/nginx-rtmp
# rtmp server open on url:  rtmp://0.0.0.0:1935/live