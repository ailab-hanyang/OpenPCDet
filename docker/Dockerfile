ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HOST_UID
ARG HOST_GID
ARG HOST_USER
ENV HOST_UID=${HOST_UID?-1000}
ENV GROUP_ID=${HOST_GID?-1000}
ENV HOST_USER=${HOST_USER}

RUN groupadd -g $HOST_GID $HOST_USER && \
    useradd -rm -u $HOST_UID -g $HOST_USER $HOST_USER && \
    usermod -aG sudo $HOST_USER && \
    echo "$HOST_USER ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# OpenPCDet Library
RUN echo "export PYTHONPATH=/home/$HOST_USER/OpenPCDet:\$PYTHONPATH" >> /home/$HOST_USER/.bashrc

# ROS1 Noetic
RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$HOST_USER/.bashrc 
# ROS2 Foxy
# RUN echo "source /opt/ros/foxy/setup.bash" >> /home/$HOST_USER/.bashrc
