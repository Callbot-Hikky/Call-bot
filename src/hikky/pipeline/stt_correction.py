"""Correction phonétique des transcriptions sur un lexique métier fermé.

Le STT (Whisper) n'a aucun mécanisme pour « renforcer » un vocabulaire : son
`initial_prompt` n'est que du contexte textuel. Résultat observé en appel réel
(2026-10-03) : « est-ce que le restaurant est halal » transcrit « à l'al »,
« à l'âge », « à l'âme » — trois appels, trois ratés sur le même mot, alors que
phonétiquement « à l'al » = /alal/ = « halal ».

Ici : une clé phonétique approximative du français (graphème → son), calculée
pour chaque mot du lexique et pour chaque fenêtre de 1 à 3 mots de la phrase ;
quand les clés coïncident (ou diffèrent d'un son pour les mots longs), la fenêtre
est remplacée par le mot du lexique. Déterministe, < 1 ms, sans dépendance.

Le motif « correction phonétique en amont, modèle en aval » est celui que la
littérature récente sur la correction d'erreurs ASR retient (jusqu'à −46 % sur les
homophones). Volontairement strict : un faux positif ferait répondre à côté.
"""

from __future__ import annotations

import re
import unicodedata

# Ce que les clients demandent au téléphone, hors créneau. Un mot ajouté ici est
# corrigé partout ; les variantes fléchies utiles sont listées explicitement.
LEXIQUE: tuple[str, ...] = (
    "halal", "casher", "végétarien", "végétarienne", "végétariens", "végan",
    "terrasse", "rooftop", "parking", "fumoir", "climatisation", "wifi",
    "poussette", "gluten", "allergène", "allergènes", "allergie", "allergies",
    "handicapé", "fauteuil", "ambiance", "privatiser", "anniversaire",
    "emporter", "livraison", "réservation",
)

# Mots-outils : une fenêtre faite uniquement de ceux-là n'est jamais corrigée
# (« à la », « il a »…). Ils peuvent en revanche faire partie d'une fenêtre plus
# large : « à l'al » = « à » + « l'al ».
_OUTILS = frozenset(
    "a à la le les l de d du des un une et ou où en on au aux il elle est es "
    "ai as c ce se sa son ses ça ca y ne pas que qui si me te vous nous je tu".split()
)

_MOT = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’]+")


def _sans_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _cle(texte: str) -> str:
    """Clé phonétique grossière du français : assez pour confondre ce qui se
    prononce pareil, pas plus. Travaille sur la suite de lettres sans espaces."""
    s = _sans_accents(texte.lower())
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""
    # Groupes consonantiques et lettres muettes.
    s = s.replace("h", "")
    s = s.replace("ph", "f").replace("ch", "S").replace("sch", "S")
    s = s.replace("qu", "k").replace("gu", "g")
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k")
    s = re.sub(r"g(?=[eiy])", "j", s)
    s = s.replace("x", "ks").replace("w", "v").replace("y", "i")
    # Voyelles composées, puis nasales (devant consonne ou en finale).
    s = s.replace("eau", "o").replace("au", "o").replace("ai", "e").replace("ei", "e")
    s = s.replace("ou", "u").replace("oi", "wa")
    s = re.sub(r"(ain|ein|in|im|un|um)(?![aeiou])", "I", s)
    s = re.sub(r"(an|am|en|em)(?![aeiou])", "A", s)
    s = re.sub(r"(on|om)(?![aeiou])", "O", s)
    # Finales muettes : -e, -es, -ent, puis une consonne finale muette courante.
    s = re.sub(r"(es|ent|e)$", "", s)
    s = re.sub(r"(?<=[aeiouAIO])[sxztd]$", "", s)
    # Lettres doublées.
    s = re.sub(r"(.)\1+", r"\1", s)
    return s


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_CLES_LEXIQUE: tuple[tuple[str, str], ...] = tuple((m, _cle(m)) for m in LEXIQUE)
# Formes déjà correctes (singulier/pluriel) : on ne les retouche jamais.
_LEX_FORMES = frozenset(
    _sans_accents(f) for m in LEXIQUE for f in (m, m + "s", m + "es")
)


def _tolerance(cle_lex: str) -> int:
    # Un son d'écart admis seulement sur les mots assez longs pour que ce soit sûr.
    return 1 if len(cle_lex) >= 4 else 0


def _correspondance(fenetre: str) -> str | None:
    cle = _cle(fenetre)
    if len(cle) < 3:
        return None
    meilleur: tuple[int, str] | None = None
    for mot, cle_lex in _CLES_LEXIQUE:
        d = _levenshtein(cle, cle_lex)
        if d <= _tolerance(cle_lex) and (meilleur is None or d < meilleur[0]):
            meilleur = (d, mot)
    return meilleur[1] if meilleur else None


def corriger_transcription(texte: str, *, detail: bool = False):
    """Remplace les fenêtres de 1 à 3 mots qui sonnent comme un mot du lexique.

    Retourne le texte corrigé ; avec `detail=True`, le couple
    (texte, [(entendu, corrigé), …]) pour la trace.
    """
    if not texte:
        return (texte, []) if detail else texte
    mots = list(_MOT.finditer(texte))
    changements: list[tuple[str, str]] = []
    sortie: list[str] = []
    pos = 0
    i = 0
    while i < len(mots):
        remplacement = None
        for taille in (3, 2, 1):
            if i + taille > len(mots):
                continue
            debut, fin = mots[i].start(), mots[i + taille - 1].end()
            fenetre = texte[debut:fin]
            tokens = [_sans_accents(m.group().lower()).replace("’", "'") for m in mots[i:i + taille]]
            # Une fenêtre faite uniquement de mots-outils ne porte pas de sens.
            if all(t.strip("'") in _OUTILS for t in tokens):
                continue
            # Un mot du lexique déjà présent tel quel (« un fumoir », « ou végan ») :
            # la fenêtre n'a rien à corriger — sinon on avalait l'article.
            if any(t.split("'")[-1] in _LEX_FORMES for t in tokens):
                break
            mot = _correspondance(fenetre)
            if mot:
                remplacement = (debut, fin, fenetre, mot, taille)
                break
        if remplacement:
            debut, fin, fenetre, mot, taille = remplacement
            if fenetre[:1].isupper():
                mot = mot[:1].upper() + mot[1:]
            sortie.append(texte[pos:debut])
            sortie.append(mot)
            changements.append((fenetre, mot))
            pos = fin
            i += taille
        else:
            i += 1
    sortie.append(texte[pos:])
    resultat = "".join(sortie)
    return (resultat, changements) if detail else resultat
