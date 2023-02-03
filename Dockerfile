FROM energyshield-carla-deps:local

ARG USER_NAME
ARG UID
ARG GID
ARG CORES

# RUN addgroup --gid ${GID} ${USER_NAME}
# RUN useradd -rm -d /home/${USER_NAME} -s /bin/bash -g ${USER_NAME} -G sudo -u ${UID} ${USER_NAME}
RUN groupmod --gid $GID ${USER_NAME} && \
    usermod --uid $UID ${USER_NAME} && \
    mkdir -p /home/${USER_NAME}/EnergyShield && \
    mv /home/${USER_NAME}/custom/models* /home/${USER_NAME}/EnergyShield/models && \
    chown -R ${UID}:${GID} /home/${USER_NAME}
RUN usermod -a -G sudo ${USER_NAME}
RUN echo "${USER_NAME}:${USER_NAME}" | chpasswd

RUN mkdir -p /media/azuredata
RUN chown -R ${UID}:${GID} /media/azuredata
RUN mkdir -p /media/azuretmp
RUN chown -R ${UID}:${GID} /media/azuretmp

# switch to unpriviledged user, and configure remote access

USER ${USER_NAME}

RUN mkdir -p /home/${USER_NAME}/.ssh
RUN mkdir -p /home/${USER_NAME}/results

RUN ssh-keygen -t rsa -q -f /home/${USER_NAME}/.ssh/id_rsa -N ""
RUN cat /home/${USER_NAME}/.ssh/id_rsa.pub >> /home/${USER_NAME}/.ssh/authorized_keys

RUN echo "export TF_CPP_MIN_LOG_LEVEL=3" >> /home/${USER_NAME}/.bashrc

COPY --chown=${UID}:${GID} . /home/${USER_NAME}/EnergyShield


USER root

# RUN mv /home/${USER_NAME}/custom/models* /home/${USER_NAME}/EnergyShield/models

# RUN chown -R ${UID}:${GID} /home/${USER_NAME}

COPY ./DockerConfig/startup.sh /usr/local/bin/startup.sh
RUN chmod 755 /usr/local/bin/startup.sh

ENTRYPOINT [ "/usr/local/bin/startup.sh" ]
