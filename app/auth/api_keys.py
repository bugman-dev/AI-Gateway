from __future__ import annotations

import secrets

from app.config import ApiKeyConfig
from app.schemas.internal import Identity


class KeyStore:
    def __init__(self, records: list[ApiKeyConfig]) -> None:
        self._records = [
            (record.key, Identity(id=record.id, allow_models=tuple(record.allow_models))) for record in records
        ]

    def lookup(self, raw_key: str) -> Identity | None:
        found: Identity | None = None
        for secret, identity in self._records:
            if secrets.compare_digest(secret, raw_key):
                found = identity
        return found
