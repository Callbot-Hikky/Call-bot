"""Pipeline téléphonique temps réel du bot Hikky (côté pod GPU).

Un appel entre par Telnyx, est écouté, compris, répondu — en flux. Chaque fichier
a UN rôle et se lit seul :

    audio.py          conversions : codecs téléphoniques, cadences, trames de 20 ms
    turn_detector.py  « le client parle / a fini / coupe le bot » — octets et durées, rien d'autre
    repair.py         que dire quand on n'a rien compris (avancer, reformuler, clore)
    call_state.py     l'état d'un appel : qui parle, énoncé gardé, répétitions, échecs
    inbound.py        la réponse à Telnyx quand le téléphone sonne (TeXML), un seul flux par appel
    speaker.py        faire parler le bot : flux TTS -> trames -> Telnyx, au bon rythme
    startup.py        démarrer et arrêter : LLM, STT/TTS/backend branchés, salutation en cache
    server.py         les trois points d'entrée HTTP/WebSocket, et la boucle de l'appel

Les constantes (seuils, durées) viennent d'appels réels et sont commentées là où
elles sont définies. L'ancien fichier monolithique `telnyx_bot.py` (477 lignes) a été supprimé
après validation du découpage par appels réels (2026-10-04).
"""
