# Asterisk containerisé pour le POC Hikky.
#
# Remplace la VM FreePBX pour les phases de dev et de démo devant les profs /
# clients. Beaucoup plus léger : ~200 Mo d'image vs ~2 Go de VM.
#
# On part de l'image communautaire `andrius/asterisk` :
#  - multi-arch (arm64 natif sur Apple Silicon, pas d'émulation Rosetta)
#  - Asterisk 22 pré-compilé avec les modules AudioSocket inclus
#    (app_audiosocket, chan_audiosocket, res_audiosocket)
#  - maintenue et régulièrement mise à jour
#
# On overlay nos configs (via templates + entrypoint qui substitue les
# variables d'env) par-dessus les défauts de l'image.
#
# Build :
#     docker compose build asterisk
# Run  :
#     docker compose up asterisk

FROM andrius/asterisk:22

# gettext-base fournit `envsubst` — utile pour substituer les variables
# d'env dans nos templates de config au démarrage.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        gettext-base \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    || (apk add --no-cache gettext 2>/dev/null || true)

# Templates avec placeholders __BOT_HOST__ / __BOT_PORT__ / etc.
# Résolus au démarrage par l'entrypoint.
COPY conf/ /etc/asterisk-templates/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ports SIP (signalisation) et RTP (media audio).
EXPOSE 5060/udp
EXPOSE 10000-10100/udp

ENTRYPOINT ["/entrypoint.sh"]
