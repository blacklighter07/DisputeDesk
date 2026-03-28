from dispute_desk.models import CaseAction, CaseObservation, EnvironmentStateModel

DisputeDeskAction = CaseAction
DisputeDeskObservation = CaseObservation
DisputeDeskState = EnvironmentStateModel

__all__ = [
    "DisputeDeskAction",
    "DisputeDeskEnv",
    "DisputeDeskObservation",
    "DisputeDeskState",
    "CaseAction",
    "CaseObservation",
    "EnvironmentStateModel",
]


def __getattr__(name: str):
    if name == "DisputeDeskEnv":
        from dispute_desk.client import DisputeDeskEnv

        return DisputeDeskEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
