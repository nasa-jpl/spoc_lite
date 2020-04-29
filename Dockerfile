FROM ros:indigo-perception

RUN apt-get update && apt-get install -y \
    python-catkin-tools
