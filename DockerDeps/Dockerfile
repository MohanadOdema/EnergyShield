FROM carlasim/carla:0.9.11


# Should only be built with user_name=carla
ENV USER_NAME=carla


ENV BASE_URL=https://us.download.nvidia.com/XFree86/Linux-x86_64
ENV DRIVER_VERSION=515.76
ENV DRIVER_BRANCH=515
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=void

USER root
WORKDIR /tmp/docker_build
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/nightly
RUN apt-get update && apt-get -y install python3.10 python3.10-dev python3.10-distutils libpython3.10-dev curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN apt-get update && \
    apt-get -y upgrade && \
    apt -y install python3-pip clang-8 lld-8 bash ninja-build zlib1g-dev libjpeg-dev libtiff-dev rsync cmake gfortran libgsl-dev libfftw3-3 libfftw3-dev libsuitesparse-dev git libgmp-dev vim emacs nano screen tmux ipython3 openssh-server sudo curl psmisc locales util-linux git-lfs kmod && \
    curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt -y install nodejs
# Install MS PowerShell
RUN wget -q "https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/packages-microsoft-prod.deb"
RUN sudo dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb
RUN sudo apt-get update && sudo apt-get install -y powershell

# Install CUDA from Ubuntu repositories (old versions)
# RUN apt -y install nvidia-cuda-toolkit

# Atempt to remove installed nvidia runtimes and replace with .run file installes
# (Cannot install from nvidia repos because deb packages try to build kernel modules)
# The following is based on: https://gitlab.com/nvidia/container-images/driver/-/blob/master/ubuntu18.04/Dockerfile
RUN apt -y clean && \
    apt update && \
    apt -y purge cuda && \
    apt -y purge nvidia-* && \
    apt -y autoremove
# The following will fail because it installs nvidia drivers from nvidia repos:
# RUN apt -y install cuda
RUN cd /tmp && \
    curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run && \
    sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run --extract-only && \
    cd NVIDIA-Linux-x86_64-$DRIVER_VERSION* && \
    ./nvidia-installer --silent \
                       --no-kernel-module \
                       --install-compat32-libs \
                       --no-nouveau-check \
                       --no-nvidia-modprobe \
                       --no-rpms \
                       --no-backup \
                       --no-check-for-alternate-installs \
                       --no-libglx-indirect \
                       --no-install-libglvnd \
                       --x-prefix=/tmp/null \
                       --x-module-path=/tmp/null \
                       --x-library-path=/tmp/null \
                       --x-sysconfig-path=/tmp/null \
                       --no-systemd && \
    mkdir -p /usr/src/nvidia-$DRIVER_VERSION && \
    mv LICENSE mkprecompiled kernel /usr/src/nvidia-$DRIVER_VERSION && \
    sed '9,${/^\(kernel\|LICENSE\)/!d}' .manifest > /usr/src/nvidia-$DRIVER_VERSION/.manifest && \
    rm -rf /tmp/*
RUN cd /tmp && \
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --toolkit --no-drm --silent && \
    echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> /etc/profile.d/cuda.sh && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> /etc/profile.d/cuda.sh && \
    chmod 644 /etc/profile.d/cuda.sh && \
    /bin/bash -c "source /etc/profile.d/cuda.sh && ldconfig" && \
    rm cuda_11.8.0_520.61.05_linux.run && \
    apt-get -y install --no-install-recommends libcudnn8 libcudnn8-dev

# Install latest neovim
# Can't be installed in 18.04
# RUN wget https://github.com/neovim/neovim/releases/download/v0.8.0/nvim-linux64.deb 
# RUN apt install ./nvim-linux64.deb
# RUN rm ./nvim-linux64.deb

RUN dpkg -r --force-depends "python3-httplib2"
RUN dpkg -r --force-depends "python3-pexpect"
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install numpy==1.22.4 tensorflow==2.9.2 tf-models-official==2.9.2 scipy Cython mpmath matplotlib onnx onnxruntime tf2onnx torch torchvision torchaudio pylint flake8 vim-vint 'python-lsp-server[all]' pylsp-mypy pyls-isort pynvim cdifflib tree_sitter gym networkx pandas pygame opencv-python tensorflow_hub distro ipython scikit_learn joblib ipykernel onnx_tf tensorflow_probability==0.17.0
RUN npm install -g vim-language-server node-gyp tree-sitter tree-sitter-cli

RUN sed -i '16i Port 5000' /etc/ssh/sshd_config
RUN /usr/bin/ssh-keygen -A
RUN sed -i -E -e 's/\s*#\s*PasswordAuthentication\s+(yes|no)/PasswordAuthentication no/' /etc/ssh/sshd_config
RUN service ssh start
EXPOSE 5000
# CMD ["/bin/bash"]

# Thsese are related to neovim; in future builds, they will be incorporated into earlier layers
RUN locale-gen en_US.UTF-8

RUN echo "#!/bin/bash" > /usr/local/bin/bk
RUN echo "(" >>  /usr/local/bin/bk
RUN echo 'echo "Date: `date`"' >>  /usr/local/bin/bk
RUN echo 'echo "Command: $*"' >>  /usr/local/bin/bk
RUN echo 'nohup "$@"' >>  /usr/local/bin/bk
RUN echo 'echo "Completed: `date`"' >>  /usr/local/bin/bk
RUN echo "" >>  /usr/local/bin/bk
RUN echo ") >>\${LOGFILE:=log.out} 2>&1 &" >>  /usr/local/bin/bk
RUN chmod 755 /usr/local/bin/bk

# Delete some groups that overlap with MacOS standard user groups
RUN delgroup --only-if-empty dialout
RUN delgroup --only-if-empty fax
RUN delgroup --only-if-empty voice



# Install neovim stuff:

# RUN mkdir -p /home/${USER_NAME}/.local/share/nvim/site/pack/packer/opt/
# RUN git clone --depth=1 https://github.com/wbthomason/packer.nvim ~/.local/share/nvim/site/pack/packer/opt/packer.nvim
# RUN mkdir -p /home/${USER_NAME}/.local/nvim
# RUN git clone --depth=1 https://github.com/jferlez/nvim-config.git /home/${USER_NAME}/.config/nvim

# WORKDIR /home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum
# RUN python3.9 posetFastCharm_numba.py

WORKDIR /home/${USER_NAME}

#RUN echo "export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum" >> /home/${USER_NAME}/.bashrc
#RUN sed -i "4i export PYTHONPATH=/home/${USER_NAME}/tools/FastBATLLNN:/home/${USER_NAME}/tools/FastBATLLNN/HyperplaneRegionEnum:/home/${USER_NAME}/tools/FastBATLLNN/TLLnet:/home/${USER_NAME}/tools/nnenum/src/nnenum" /home/${USER_NAME}/.bashrc
RUN echo "export TERM=xterm-256color" >> /home/${USER_NAME}/.bashrc
RUN echo "export COLORTERM=truecolor" >> /home/${USER_NAME}/.bashrc
RUN echo "export TERM_PROGRAM=iTerm2.app" >> /home/${USER_NAME}/.bashrc
RUN echo "set-option -gs default-terminal \"tmux-256color\" # Optional" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-option -gas terminal-overrides \"*:Tc\"" >> /home/${USER_NAME}/.tmux.conf
RUN echo "set-option -gas terminal-overrides \"*:RGB\"" >> /home/${USER_NAME}/.tmux.conf

RUN git clone https://github.com/carla-simulator/carla carla
RUN cd /home/${USER_NAME}/carla && git checkout 0.9.11
COPY --chown=${USER_NAME} ./DockerConfig/Setup.sh /home/${USER_NAME}/carla/Util/BuildTools/Setup.sh

USER root
RUN update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-8/bin/clang++ 180
RUN update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-8/bin/clang 180

# Save EnergyShield clone for non-deps image
# COPY --chown=${UID}:${GID} ./EnergyShield /home/${USER_NAME}/EnergyShield/EnergyShield
COPY --chown=${USER_NAME} ./TensorFlow /home/${USER_NAME}/EnergyShield/TensorFlow
COPY --chown=${USER_NAME} ./models /home/${USER_NAME}/EnergyShield/models

RUN cd /home/${USER_NAME}/EnergyShield/TensorFlow/models && python3.10 -m pip install -e research

USER root
RUN cd /home/${USER_NAME}/carla && make PythonAPI ARGS="--python-version=3.10"


RUN cd /home/${USER_NAME}/carla/PythonAPI/carla && python3.10 ./setup.py install
RUN rm -rf /home/${USER_NAME}/carla

RUN chsh -s /bin/bash ${USER_NAME}

RUN chown -R ${USER_NAME} /home/${USER_NAME}/

# Remove these from dependencies image:
# COPY ./DockerConfig/startup.sh /usr/local/bin/startup.sh
# RUN chmod 755 /usr/local/bin/startup.sh

# ENTRYPOINT [ "/usr/local/bin/startup.sh" ]
