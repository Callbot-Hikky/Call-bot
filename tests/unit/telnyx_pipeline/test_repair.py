"""Que dire quand on n'a rien compris : avancer, reformuler, puis clore."""

from telnyx_pipeline.repair import MAX_ATTEMPTS, repair_reply


def test_premier_echec_on_avance_sur_la_question_utile():
    text, end = repair_reply(
        1, awaiting_confirmation=False, next_question="Pour combien de personnes ?", recap=None
    )
    assert text.startswith("Je vous entends mal.") and "combien" in text and not end


def test_deuxieme_echec_on_reformule():
    text, end = repair_reply(
        2, awaiting_confirmation=False, next_question="À quelle heure ?", recap=None
    )
    assert "ligne" in text.lower() and "heure" in text and not end


def test_troisieme_echec_on_clot_poliment():
    text, end = repair_reply(
        MAX_ATTEMPTS, awaiting_confirmation=False, next_question="À quelle heure ?", recap=None
    )
    assert end and "rappeler" in text.lower()


def test_en_attente_de_confirmation_on_redemande_oui_ou_non():
    text, end = repair_reply(1, awaiting_confirmation=True, next_question=None, recap="Récap ?")
    assert "oui" in text.lower() and not end


def test_tout_est_connu_on_redit_le_recapitulatif():
    text, _ = repair_reply(
        1,
        awaiting_confirmation=False,
        next_question=None,
        recap="Je récapitule : deux personnes. C'est bien cela ?",
    )
    assert text.endswith("C'est bien cela ?")
