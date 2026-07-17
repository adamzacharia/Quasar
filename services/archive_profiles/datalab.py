"""Data Lab archive profile — validated from the registry adapter.

All authoring lives in ``services/datalab_registry.py`` (TABLE_PROFILE_INFO +
the _PROFILE_* constants + ``to_profile()``), keeping the registry the single
source of Data Lab knowledge. This module only validates the plain dict into
the typed ArchiveProfile, so the registry never imports the profile package
(duel task-81b3b73-2145, DX-06).
"""

from __future__ import annotations

from services import datalab_registry
from services.archive_profiles.schema import ArchiveProfile

PROFILE: ArchiveProfile = ArchiveProfile.model_validate(datalab_registry.to_profile())
