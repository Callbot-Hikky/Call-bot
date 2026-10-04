"""Le pipeline téléphonique vit hors de `src/` (il tourne sur le pod, pas dans le
paquet `hikky`). On l'ajoute au chemin d'import pour le tester ici."""

import sys
from pathlib import Path

SERVICES = Path(__file__).resolve().parents[3] / "runpod_deploy" / "services"
if str(SERVICES) not in sys.path:
    sys.path.insert(0, str(SERVICES))
