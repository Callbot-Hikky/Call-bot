from hikky.domain.outcomes import CallOutcome
from tests.replay.conversation_replay import (
    ConversationReplay,
    ExpectBackCalled,
    ExpectOutcome,
    UserSays,
)


async def test_three_turns_no_progress_triggers_callback():
    replay = (
        ConversationReplay()
        .with_default_context()
        .with_llm_replies(
            ["Pardon ?", "Je n'ai pas compris.", "Pouvez-vous répéter ?"]
        )
    )
    await replay.run(
        [
            UserSays("euh..."),
            UserSays("hum..."),
            UserSays("..."),
            ExpectBackCalled("create_callback_request"),
            ExpectOutcome(CallOutcome.CALLBACK_REQUESTED),
        ],
        customer_phone="+33600000000",
    )
