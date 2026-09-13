#!/bin/sh
# Substitue les placeholders __BOT_HOST__ / __BOT_PORT__ dans les configs
# templates puis démarre Asterisk au premier plan (logs vers stdout).
#
# Défauts : Docker Desktop Mac/Windows expose l'hôte via `host.docker.internal`.
# Sur Linux, docker-compose ajoute cette entrée via `extra_hosts`.
set -eu

: "${BOT_HOST:=host.docker.internal}"
: "${BOT_PORT:=6666}"
: "${SIP_EXTENSION_PASSWORD:=1000}"

echo "[entrypoint] BOT_HOST=$BOT_HOST BOT_PORT=$BOT_PORT"

# Le chemin des modules dépend de l'architecture (x86_64-linux-gnu sur Intel/AMD,
# aarch64-linux-gnu sur Raspberry Pi 64 bits). On le détecte au lieu de le figer.
MODDIR=""
for candidate in /usr/lib/*/asterisk/modules /usr/lib/asterisk/modules; do
    if [ -d "$candidate" ]; then MODDIR="$candidate"; break; fi
done
if [ -z "$MODDIR" ]; then
    echo "[entrypoint] ERREUR: repertoire des modules Asterisk introuvable" >&2
    exit 1
fi
echo "[entrypoint] MODDIR=$MODDIR ($(uname -m))"

if [ ! -f "$MODDIR/app_audiosocket.so" ]; then
    echo "[entrypoint] ERREUR: app_audiosocket.so absent de $MODDIR" >&2
    echo "[entrypoint] l'appel vers le bot ne pourra pas fonctionner" >&2
    exit 1
fi

mkdir -p /etc/asterisk

for tpl in /etc/asterisk-templates/*.conf; do
    name=$(basename "$tpl")
    sed \
        -e "s|__BOT_HOST__|$BOT_HOST|g" \
        -e "s|__BOT_PORT__|$BOT_PORT|g" \
        -e "s|__SIP_EXTENSION_PASSWORD__|$SIP_EXTENSION_PASSWORD|g" \
        -e "s|__MODDIR__|$MODDIR|g" \
        "$tpl" > "/etc/asterisk/$name"
done

chown -R asterisk:asterisk /etc/asterisk /var/lib/asterisk /var/spool/asterisk \
    /var/log/asterisk /var/run/asterisk 2>/dev/null || true

# Foreground + verbose pour que `docker logs` soit utile.
exec /usr/sbin/asterisk -f -vvv -U asterisk -G asterisk
