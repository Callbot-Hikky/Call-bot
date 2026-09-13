"""Le code decide quoi dire, le modele decide comment le dire.

L'aiguilleur a corrige un vrai probleme : le modele perdait l'etat de la
reservation et redemandait trois fois la meme chose. Mais le remede a
retire au modele la parole *entierement* — chaque phrase du bot est
devenue une chaine figee, piochee dans trois variantes. Sur un appel
reel, le modele n'etait appele pour parler aucune fois.

D'ou l'impression de parler a un serveur vocal.

Le partage correct :

    le code   : quel slot manque, quelle reservation confirmer, quoi
                enregistrer — l'etat, qu'il ne doit jamais lacher
    le modele : la formulation, qui tient compte de ce que le client
                vient de dire

Le repli sur la phrase figee reste indispensable : au telephone, mieux
vaut une phrase un peu raide que trois secondes de silence.
"""

import asyncio

from hikky.domain.phraseur import Intention, Phraseur


class _ModeleFidele:
    """Modele qui formule correctement."""

    def __init__(self, reponse: str = "Très bien, et pour quel jour ?") -> None:
        self.reponse = reponse
        self.prompts: list[str] = []

    async def complete(self, messages: list[dict[str, str]]) -> str:
        self.prompts.append(messages[-1]["content"])
        return self.reponse


class _ModeleMuet:
    """Modele indisponible : leve, comme le ferait un timeout."""

    async def complete(self, messages: list[dict[str, str]]) -> str:
        raise TimeoutError("le modele n'a pas repondu")


class _ModeleLent:
    async def complete(self, messages: list[dict[str, str]]) -> str:
        await asyncio.sleep(5)
        return "trop tard"


def _intention(slot: str = "date") -> Intention:
    return Intention(
        besoin="demander_slot",
        slot=slot,
        repli="Pour quel jour souhaitez-vous réserver ?",
        derniere_phrase_client="Bonjour, je voudrais une table pour deux.",
    )


# ── Le modele reprend la parole ─────────────────────────────────────────


async def test_la_formulation_vient_du_modele():
    modele = _ModeleFidele("Avec plaisir. Vous souhaitez venir quel jour ?")
    phraseur = Phraseur(modele)

    dit = await phraseur.formuler(_intention())

    assert dit == "Avec plaisir. Vous souhaitez venir quel jour ?"


async def test_le_modele_recoit_ce_que_le_client_vient_de_dire():
    """Sans cela il ne peut pas enchainer — il recite."""
    modele = _ModeleFidele()
    await Phraseur(modele).formuler(_intention())

    prompt = modele.prompts[0]
    assert "table pour deux" in prompt


async def test_le_modele_sait_quelle_information_demander():
    modele = _ModeleFidele()
    await Phraseur(modele).formuler(_intention(slot="party_size"))

    prompt = modele.prompts[0]
    assert "party_size" in prompt or "combien" in prompt.lower()


# ── Le repli protege l'appel ────────────────────────────────────────────


async def test_un_modele_indisponible_ne_coupe_pas_l_appel():
    dit = await Phraseur(_ModeleMuet()).formuler(_intention())
    assert dit == "Pour quel jour souhaitez-vous réserver ?"


async def test_un_modele_trop_lent_est_abandonne():
    """Au telephone, trois secondes de silence sonnent comme un appel coupe."""
    phraseur = Phraseur(_ModeleLent(), timeout_secondes=0.2)
    dit = await phraseur.formuler(_intention())
    assert dit == "Pour quel jour souhaitez-vous réserver ?"


async def test_une_reponse_vide_bascule_sur_le_repli():
    dit = await Phraseur(_ModeleFidele("   ")).formuler(_intention())
    assert dit == "Pour quel jour souhaitez-vous réserver ?"


# ── Le modele ne doit pas deraper ───────────────────────────────────────


async def test_une_formulation_a_rallonge_est_refusee():
    """Le modele part parfois en tirade. Au telephone c'est insupportable."""
    tirade = "Alors, " + "je vous explique en detail nos differentes formules. " * 12
    dit = await Phraseur(_ModeleFidele(tirade)).formuler(_intention())
    assert dit == "Pour quel jour souhaitez-vous réserver ?"


async def test_le_modele_ne_peut_pas_inventer_une_confirmation():
    """Le recapitulatif reste ecrit par le code : il engage le restaurant.

    Le modele a deja annonce des reservations qui n'existaient pas. Sur
    les intentions qui engagent, sa formulation n'est pas sollicitee.
    """
    modele = _ModeleFidele("C'est reserve, a demain !")
    intention = Intention(
        besoin="recapituler",
        slot=None,
        repli="Je récapitule : 2 personnes demain à midi. C'est bien cela ?",
        derniere_phrase_client="oui",
    )

    dit = await Phraseur(modele).formuler(intention)

    assert dit == "Je récapitule : 2 personnes demain à midi. C'est bien cela ?"
    assert modele.prompts == [], "le modele ne doit pas etre consulte ici"
