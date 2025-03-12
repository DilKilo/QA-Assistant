#!/bin/bash
set -e

# environment variables
USER_NAME=chroma
CHROMA_VERSION=0.6.3
PERSIST_DIRECTORY=/data
CHROMA_OPEN_TELEMETRY__ENDPOINT=
CHROMA_OPEN_TELEMETRY__SERVICE_NAME=
OTEL_EXPORTER_OTLP_HEADERS={}
GCSFUSE_BUCKET=chromadb-vectors-storage

# user

useradd -m -s /bin/bash $USER_NAME

apt-get update

# docker
apt-get install -y docker.io

usermod -aG docker $USER_NAME

curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

systemctl enable docker
systemctl start docker

mkdir -p /home/$USER_NAME/config
mkdir -p /home/$USER_NAME/chroma_data
curl -o /home/$USER_NAME/docker-compose.yml https://s3.amazonaws.com/public.trychroma.com/cloudformation/assets/docker-compose.yml

sed -i "s/CHROMA_VERSION/$CHROMA_VERSION/g" /home/$USER_NAME/docker-compose.yml
sed -i 's#- index_data:/index_data#- /home/chroma/chroma_data:/data#g' /home/$USER_NAME/docker-compose.yml

cat <<EOF > /home/$USER_NAME/.env
PERSIST_DIRECTORY=$PERSIST_DIRECTORY
CHROMA_OPEN_TELEMETRY__ENDPOINT=$CHROMA_OPEN_TELEMETRY__ENDPOINT
CHROMA_OPEN_TELEMETRY__SERVICE_NAME=$CHROMA_OPEN_TELEMETRY__SERVICE_NAME
OTEL_EXPORTER_OTLP_HEADERS=$OTEL_EXPORTER_OTLP_HEADERS
EOF

chown $USER_NAME:$USER_NAME /home/$USER_NAME/.env /home/$USER_NAME/docker-compose.yml /home/$USER_NAME/chroma_data

# gcsfuse

apt-get update
apt-get install -y gnupg lsb-release

export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
apt-get update
apt-get install -y gcsfuse

# mount bucket
USER_UID=$(id -u $USER_NAME)
USER_GID=$(id -g $USER_NAME)
gcsfuse --uid=$USER_UID --gid=$USER_GID --implicit-dirs -o rw,allow_other $GCSFUSE_BUCKET /home/$USER_NAME/chroma_data

# run docker-compose
cd /home/$USER_NAME
sudo -u $USER_NAME docker-compose up -d