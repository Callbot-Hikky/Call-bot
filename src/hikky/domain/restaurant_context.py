from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpeningHours(BaseModel):
    model_config = ConfigDict(frozen=True)

    weekday: int = Field(ge=0, le=6, description="0 = lundi, 6 = dimanche")
    opens: time
    closes: time


class RestaurantRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_group_size: int = Field(gt=0, default=12)
    reservation_duration_minutes: int = Field(gt=0, default=90)


class RestaurantContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    greeting: str
    opening_hours: list[OpeningHours]
    total_capacity: int = Field(ge=0)
    rules: RestaurantRules
    transfer_number: str | None = None
    fallback_message: str

    @field_validator("opening_hours")
    @classmethod
    def at_least_one_opening(cls, v: list[OpeningHours]) -> list[OpeningHours]:
        if not v:
            raise ValueError("opening_hours must not be empty")
        return v

    def is_open_at(self, moment: datetime) -> bool:
        for slot in self.opening_hours:
            if slot.weekday == moment.weekday():
                if slot.opens <= moment.time() <= slot.closes:
                    return True
        return False
