"""Dire oui autrement que par « oui ».

Sur un appel reel, le client a confirme par « Ouais, a peu pres » puis
« Affirmatif » ; aucun des deux n'etait dans la liste de mots-cles. Le
bot a redemande confirmation trois fois de suite.

Une liste de mots-cles restera toujours incomplete — mais l'elargir
couvre le parler courant sans ajouter la moindre latence, ce qui compte
quand quelqu'un attend au bout du fil.
"""

from hikky.domain.question_router import is_agreement, is_refusal


class TestAccord:
    def test_les_formes_courantes(self):
        for phrase in (
            "oui",
            "oui, je confirme",
            "c'est ça",
            "exact",
            "tout à fait",
            "parfait",
            "d'accord",
        ):
            assert is_agreement(phrase), phrase

    def test_le_parler_courant(self):
        """Ce que les gens disent vraiment au telephone."""
        for phrase in (
            "ouais",
            "ouais, à peu près",
            "affirmatif",
            "c'est bon",
            "ça marche",
            "ca me va",
            "allez-y",
            "impeccable",
            "très bien",
            "nickel",
            "yes",
            "absolument",
            "bien sûr",
        ):
            assert is_agreement(phrase), phrase


class TestRefus:
    def test_les_formes_courantes(self):
        for phrase in ("non", "pas du tout", "c'est faux", "annulez"):
            assert is_refusal(phrase), phrase

    def test_le_parler_courant(self):
        for phrase in (
            "nan",
            "non pas vraiment",
            "négatif",
            "plutôt pas",
        ):
            assert is_refusal(phrase), phrase
        # « Je préfère changer » n'est plus un refus mais une CORRECTION : pendant
        # la confirmation, les deux mènent au même endroit (on corrige), mais hors
        # confirmation « changer » ne doit rien refuser.
        from hikky.domain.question_router import is_correction
        assert is_correction("je préfère changer") and not is_refusal("je préfère changer")


class TestAmbiguite:
    def test_un_refus_prime_sur_un_accord(self):
        """« Non, c'est pas ça » contient « c'est ça » : le refus gagne."""
        assert is_refusal("non, c'est pas ça")
        assert not is_agreement("non, c'est pas ça")

    def test_un_accord_ne_se_declenche_pas_sur_un_fragment(self):
        """« Grégoire » contient « oi » mais pas « oui » — et « Ouissem »
        commence par « oui ». Un nom ne vaut pas confirmation."""
        assert not is_agreement("Ouissem")
        assert not is_agreement("au nom de Louison")

    def test_une_phrase_neutre_n_est_ni_l_un_ni_l_autre(self):
        for phrase in ("demain à midi", "quatre personnes", "au nom de Dupont"):
            assert not is_agreement(phrase), phrase
            assert not is_refusal(phrase), phrase
