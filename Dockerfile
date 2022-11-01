FROM energyshield-carla-deps:local

ARG USER_NAME
ARG UID
ARG GID
ARG CORES


COPY --chown=${UID}:${GID} ./EnergyShield /home/${USER_NAME}/EnergyShield/EnergyShield

RUN chsh -s /bin/bash ${USER_NAME}

RUN chown -R ${UID}:${GID} /home/${USER_NAME}/

COPY ./DockerConfig/startup.sh /usr/local/bin/startup.sh
RUN chmod 755 /usr/local/bin/startup.sh

ENTRYPOINT [ "/usr/local/bin/startup.sh" ]
