from dataclasses import dataclass

from hikky.domain.outcomes import CallOutcome
from hikky.domain.restaurant_context import RestaurantContext

NO_PROGRESS_THRESHOLD = 3

# Marqueurs d'une demande de répétition. Ces tours n'apportent aucun slot
# mais signalent un problème du BOT (réponse inaudible, question mal
# formulée), pas un client bloqué. Les compter comme absence de progrès
# revient à raccrocher au nez de quelqu'un qui coopère — constaté en
# appel réel : « Hein ? T'as dit quoi ? Tu peux répéter ? » a contribué
# au déclenchement du repli.
_CLARIFICATION_MARKERS = (
    "répéter",
    "repeter",
    "pas compris",
    "pas entendu",
    "comment",
    "pardon",
    "hein",
    "quoi",
    "plus fort",
    "redis",
)


# Marqueurs d'une contestation : le client dément une information que le
# bot a retenue. C'est le signal le plus fort qu'il existe — le bot s'est
# trompé — et le compter comme absence de progrès revient à raccrocher au
# nez de quelqu'un qui essaie de corriger. Constaté en appel réel.
_CORRECTION_MARKERS = (
    "jamais dit",
    "j'ai pas dit",
    "ai pas dit",
    "c'est pas mon",
    "n'est pas mon",
    "pas mon nom",
    "qui est",
    "c'est faux",
    "pas ça",
    "pas ca",
    "me suis pas",
    "je m'appelle pas",
    "appelle pas",
)


def is_correction(text: str) -> bool:
    """Vrai si le client conteste une information retenue par le bot."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _CORRECTION_MARKERS)


def is_clarification_request(text: str) -> bool:
    """Vrai si le client demande simplement de répéter ou reformuler."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _CLARIFICATION_MARKERS)


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    outcome: CallOutcome
    reason: str
    transfer_destination: str | None = None


class FallbackPolicy:
    def decide(
        self,
        ctx: RestaurantContext,
        *,
        consecutive_no_progress_turns: int,
        user_requested_human: bool,
        group_size: int | None,
    ) -> FallbackDecision | None:
        if user_requested_human:
            if ctx.transfer_number:
                return FallbackDecision(
                    outcome=CallOutcome.TRANSFERRED,
                    reason="human_requested",
                    transfer_destination=ctx.transfer_number,
                )
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED,
                reason="human_requested_no_transfer_configured",
            )

        if group_size is not None and group_size > ctx.rules.max_group_size:
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED, reason="oversized_group"
            )

        if consecutive_no_progress_turns >= NO_PROGRESS_THRESHOLD:
            return FallbackDecision(
                outcome=CallOutcome.CALLBACK_REQUESTED, reason="no_progress"
            )

        return None
