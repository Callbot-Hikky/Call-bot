import pytest

from hikky.ports.call_log import CallLogPort
from hikky.ports.language_model import LanguageModelPort
from hikky.ports.notification import NotificationPort
from hikky.ports.reservation import ReservationPort
from hikky.ports.restaurant_context import RestaurantContextPort
from hikky.ports.speech_recognition import SpeechRecognitionPort
from hikky.ports.speech_synthesis import SpeechSynthesisPort
from hikky.ports.telephony import TelephonyPort


@pytest.mark.parametrize(
    "port_cls",
    [
        CallLogPort,
        LanguageModelPort,
        NotificationPort,
        ReservationPort,
        RestaurantContextPort,
        SpeechRecognitionPort,
        SpeechSynthesisPort,
        TelephonyPort,
    ],
)
def test_port_is_abstract_and_cannot_be_instantiated(port_cls):
    with pytest.raises(TypeError):
        port_cls()
