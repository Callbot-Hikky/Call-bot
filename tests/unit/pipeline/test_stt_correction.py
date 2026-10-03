"""Correction phonétique des transcriptions sur le lexique métier.

Appels réels du 2026-10-03 : « est-ce que le restaurant est halal » transcrit
« à l'al » (3 appels), « à l'âge », « à l'âme ». Phonétiquement « à l'al » = /alal/
= « halal » : distance zéro. Un lexique fermé de quelques dizaines de mots suffit.
"""

import pytest

from hikky.pipeline.stt_correction import corriger_transcription


@pytest.mark.parametrize(
    "entendu, attendu",
    [
        ("Oui, bonjour, j'aimerais savoir si le restaurant est à l'al.",
         "Oui, bonjour, j'aimerais savoir si le restaurant est halal."),
        ("je voudrais savoir si le restaurant est à l'âge.",
         "je voudrais savoir si le restaurant est halal."),
        ("je voudrais savoir si le restaurant est à l'âme.",
         "je voudrais savoir si le restaurant est halal."),
        ("Est-ce que vous avez une terrace ?", "Est-ce que vous avez une terrasse ?"),
        ("vous êtes cacher ?", "vous êtes casher ?"),
    ],
)
def test_les_mots_du_lexique_mal_entendus_sont_retablis(entendu, attendu):
    assert corriger_transcription(entendu) == attendu


@pytest.mark.parametrize(
    "texte",
    [
        "Je voudrais réserver une table pour deux pour demain.",
        "À quelle heure vous ouvrez ?",
        "Pour quatre personnes à vingt heures, au nom de Royal.",
        "Est-ce que le restaurant propose du halal ?",   # déjà juste : inchangé
        "Il a l'air sympa votre restaurant.",            # « a l'air » ≠ halal
    ],
)
def test_une_phrase_correcte_n_est_pas_touchee(texte):
    assert corriger_transcription(texte) == texte


def test_la_correction_est_tracee():
    texte, changements = corriger_transcription("le restaurant est à l'al ?", detail=True)
    assert texte == "le restaurant est halal ?"
    assert changements == [("à l'al", "halal")]
