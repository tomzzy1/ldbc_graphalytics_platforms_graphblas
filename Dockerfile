FROM zhongbozhu/ece508-cuda-graph-analytics:latest
LABEL maintainer=zhongbo2@illinois.edu

COPY ./rai /usr/bin
RUN apt-get update && apt-get install -y --no-install-recommends vim git 
RUN mkdir /rai-submit

RUN cd /cuda-graph-analytics/ldbc_graphalytics && mvn install && \
    cd ../ldbc_graphalytics_platforms_graphblas && \
    /bin/bash scripts/init.sh /cuda-graph-analytics/example-data-sets/graphs /cuda-graph-analytics/example-data-sets/matrices 

WORKDIR /rai-submit

# docker build -t ece508-cuda-graph-analytics-rai .
# or you can do: docker pull zhongbozhu/ece508-cuda-graph-analytics-rai:latest
# I already have the image on my local disk (as well as docker hub)
# docker run -it --gpus all -v "C:\Users\<NAME>\<DIR>\ECE508-final-ldbc-graph-cuda:/rai-submit" zhongbozhu/ece508-cuda-graph-analytics-rai /bin/bash
# add your rai_profile