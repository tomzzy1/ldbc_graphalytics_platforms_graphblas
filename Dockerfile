FROM zhongbozhu/ece508-cuda-graph-analytics:latest
LABEL maintainer=zhongbo2@illinois.edu

COPY ./rai /usr/bin
RUN apt-get update && apt-get install -y --no-install-recommends vim git zstd
RUN mkdir /rai-submit

# copy datasets into the image for rai
COPY small-data-sets/graphs/datagen-7_5-fb.v small-data-sets/graphs/datagen-7_5-fb.e small-data-sets/graphs/datagen-7_5-fb.properties small-data-sets/graphs/datagen-7_5-fb-CDLP /cuda-graph-analytics/example-data-sets/graphs/
COPY small-data-sets/graphs/datagen-7_6-fb.v small-data-sets/graphs/datagen-7_6-fb.e small-data-sets/graphs/datagen-7_6-fb.properties small-data-sets/graphs/datagen-7_6-fb-CDLP /cuda-graph-analytics/example-data-sets/graphs/
COPY small-data-sets/graphs/datagen-7_7-zf.v small-data-sets/graphs/datagen-7_7-zf.e small-data-sets/graphs/datagen-7_7-zf.properties small-data-sets/graphs/datagen-7_7-zf-CDLP /cuda-graph-analytics/example-data-sets/graphs/
COPY small-data-sets/graphs/datagen-7_8-zf.v small-data-sets/graphs/datagen-7_8-zf.e small-data-sets/graphs/datagen-7_8-zf.properties small-data-sets/graphs/datagen-7_8-zf-CDLP /cuda-graph-analytics/example-data-sets/graphs/
COPY small-data-sets/graphs/datagen-7_9-fb.v small-data-sets/graphs/datagen-7_9-fb.e small-data-sets/graphs/datagen-7_9-fb.properties small-data-sets/graphs/datagen-7_9-fb-CDLP /cuda-graph-analytics/example-data-sets/graphs/

RUN cd /cuda-graph-analytics/ldbc_graphalytics && mvn install && \
    cd ../ldbc_graphalytics_platforms_graphblas && \
    /bin/bash scripts/init-for-testing.sh /cuda-graph-analytics/example-data-sets/graphs /cuda-graph-analytics/example-data-sets/matrices 

WORKDIR /rai-submit

# docker build -t ece508-cuda-graph-analytics-rai .
# or you can do: docker pull zhongbozhu/ece508-cuda-graph-analytics-rai:latest
# I already have the image on my local disk (as well as docker hub)
# docker run -it --gpus all -v "C:\Users\<NAME>\<DIR>\ECE508-final-ldbc-graph-cuda:/rai-submit" zhongbozhu/ece508-cuda-graph-analytics-rai /bin/bash
# add your rai_profile

# for image used in rai, I added datasets inside of it, so you don't need to copy datasets into the container
# get image: docker pull zhongbozhu/ece508-cuda-graph-analytics-rai:rai-latest

# re-init by yourself:
# scripts/init-for-testing.sh /cuda-graph-analytics/example-data-sets/graphs /cuda-graph-analytics/example-data-sets/matrices 