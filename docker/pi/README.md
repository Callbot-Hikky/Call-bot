# Déploiement Asterisk sur Raspberry Pi 5

Architecture visée :

```
Zoiper (téléphone, même wifi)
    │  SIP/RTP UDP
    ▼
Asterisk (Docker sur le Pi, network_mode host)
    │  AudioSocket TCP vers 127.0.0.1:6666
    ▼
Tunnel SSH (systemd, hikky-tunnel.service)
    │
    ▼
Bot Hikky (pod RunPod GPU, port 6666)
```

Seul le tunnel sort sur internet. Le SIP reste confiné au réseau local :
rien n'est exposé publiquement, donc pas de brute-force SIP à craindre.

## Prérequis

- Raspberry Pi OS **64 bits** (`uname -m` doit répondre `aarch64` ;
  l'image Asterisk n'existe pas en armv7).
- Docker + plugin compose installés.

## 1. Clé SSH du Pi

Le tunnel s'authentifie auprès de RunPod avec une clé propre au Pi :

```bash
ssh-keygen -t ed25519 -f /root/.ssh/id_ed25519_runpod -N "" -C "hikky-pi"
cat /root/.ssh/id_ed25519_runpod.pub
```

Colle cette clé dans https://www.runpod.io/console/user/settings
(section *SSH Public Keys*).

⚠️ **RunPod injecte les clés à la création du pod, pas à son démarrage.**
Si le pod existe déjà, il faut le recréer pour qu'il prenne la clé — un
`stop`/`start` ne suffit pas.

## 2. Configurer le tunnel

Le port SSH public du pod **change à chaque redémarrage**. Il est donc
sorti dans un fichier d'environnement plutôt que figé dans l'unit systemd.

Récupérer l'endpoint courant depuis le Mac :

```bash
runpodctl get pod <POD_ID> --allfields | grep -oE '[0-9.]+:[0-9]+->22'
```

Puis sur le Pi :

```bash
cat > /etc/hikky-tunnel.env <<'EOF'
POD_HOST=194.14.47.19
POD_PORT=23257
POD_USER=root
SSH_KEY=/root/.ssh/id_ed25519_runpod
EOF
chmod 600 /etc/hikky-tunnel.env

cp hikky-tunnel.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now hikky-tunnel
systemctl status hikky-tunnel
```

Après chaque redémarrage du pod, mettre à jour `POD_PORT` puis :

```bash
systemctl restart hikky-tunnel
```

Vérifier que le tunnel est vivant :

```bash
ss -lntp | grep 6666
```

## 3. Lancer Asterisk

```bash
cd docker
docker compose -f docker-compose.pi.yml up --build -d
docker logs -f hikky-asterisk
```

Au démarrage, l'entrypoint affiche l'architecture détectée et le chemin
des modules — il refuse de démarrer si `app_audiosocket.so` est absent,
plutôt que de laisser Asterisk tourner sans pouvoir joindre le bot.

## 4. Configurer Zoiper

Sur le téléphone, connecté au **même wifi que le Pi** :

| Champ | Valeur |
|---|---|
| Serveur / domaine | IP LAN du Pi (`hostname -I` sur le Pi) |
| Utilisateur | `1000` |
| Mot de passe | valeur de `SIP_EXTENSION_PASSWORD` (défaut `1000`) |
| Transport | UDP |

Puis composer :

- `*43` → echo test Asterisk. Tu t'entends. Valide le SIP/RTP **sans** le bot.
- `2000` → appel du bot Hikky.

Toujours tester `*43` en premier : s'il échoue, le problème est réseau
(wifi, isolation client) et non applicatif.

## Dépannage

**`*43` ne marche pas** — beaucoup de wifi d'établissement activent
l'isolation client, qui bloque le trafic entre appareils du même réseau.
Contournement : faire monter au Pi son propre point d'accès, ou connecter
Pi et téléphone au partage de connexion d'un téléphone.

**`2000` sonne mais silence** — le tunnel est probablement coupé :
`systemctl status hikky-tunnel` et `ss -lntp | grep 6666`. Vérifier que
`POD_PORT` correspond toujours à l'endpoint courant du pod.

**`2000` raccroche immédiatement** — le bot n'écoute pas côté pod.
Vérifier ses logs sur le pod.
