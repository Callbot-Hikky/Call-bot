"""La premiere phrase de l'appel.

Elle doit faire trois choses, dans cet ordre :

1. situer l'appelant — il vient de composer un numero, il veut savoir
   qu'il est au bon endroit ;
2. dire honnetement que personne n'est disponible en salle, et que
   c'est un assistant vocal qui repond. Le decouvrir en cours de
   conversation met le client mal a l'aise ;
3. inviter a parler, chaleureusement.

Contrainte qui prime sur le reste : la longueur. XTTS synthetise a peu
pres 6 caracteres par 100 ms ; une phrase d'accueil bavarde fait
patienter l'appelant avant meme qu'il ait pu dire un mot.
"""

from hikky.adapters.back.backend_restaurant_context import _to_context

LIMITE_CARACTERES = 200


def _accueil(nom: str = "Le Bistrot du Coin") -> str:
    return _to_context({"restaurant": {"id": "r-1", "name": nom}}).greeting


def test_l_accueil_nomme_le_restaurant():
    """L'appelant doit savoir qu'il est au bon endroit."""
    assert "Le Bistrot du Coin" in _accueil()


def test_l_accueil_annonce_que_personne_n_est_disponible():
    accueil = _accueil().lower()
    assert any(
        marqueur in accueil
        for marqueur in ("disponible", "occupé", "occupe", "personne")
    ), accueil


def test_l_accueil_annonce_un_assistant_vocal():
    """Le client doit savoir a qui il parle des la premiere seconde."""
    accueil = _accueil().lower()
    assert any(
        marqueur in accueil for marqueur in ("assistant", "vocal", "automatique")
    ), accueil


def test_l_accueil_invite_a_parler():
    """Sans invitation claire, l'appelant reste silencieux et attend."""
    assert _accueil().rstrip().endswith(("?", "!"))


def test_l_accueil_reste_court():
    """Une phrase bavarde fait patienter avant meme le premier mot."""
    accueil = _accueil()
    assert len(accueil) <= LIMITE_CARACTERES, f"{len(accueil)} caracteres"


def test_un_restaurant_sans_nom_reste_accueillant():
    """Le backend peut renvoyer un nom vide : l'accueil doit tenir debout."""
    accueil = _to_context({"restaurant": {"id": "r-1", "name": None}}).greeting
    assert "None" not in accueil
    assert "notre restaurant" in accueil
