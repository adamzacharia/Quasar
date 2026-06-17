"""Shared admin access helpers."""

from __future__ import annotations

import os
from typing import Mapping, Optional


DEFAULT_ADMIN_EMAILS = {
    "1@1",
}


def configured_admin_emails() -> set[str]:
    """Return local-dev and environment-configured admin emails."""
    env_emails = {
        email.strip().lower()
        for email in os.getenv("ADMIN_EMAILS", "").split(",")
        if email.strip()
    }
    return DEFAULT_ADMIN_EMAILS | env_emails


def configured_quota_exempt_emails() -> set[str]:
    """Return emails allowed to bypass platform token quotas."""
    env_emails = {
        email.strip().lower()
        for email in os.getenv("QUASAR_TOKEN_LIMIT_EXEMPT_EMAILS", "").split(",")
        if email.strip()
    }
    return env_emails | configured_admin_emails()


def is_admin_email(email: Optional[str]) -> bool:
    """Return True when the email is allowed through admin gates."""
    if not email:
        return False
    return email.strip().lower() in configured_admin_emails()


def is_quota_exempt_email(email: Optional[str]) -> bool:
    """Return True when the email is allowed to bypass token quotas."""
    if not email:
        return False
    return email.strip().lower() in configured_quota_exempt_emails()


def is_admin_user(user: Optional[Mapping[str, object]]) -> bool:
    """Return True for JWT payloads or user dicts with an admin email."""
    if not user:
        return False
    email = str(user.get("email") or user.get("username") or "").strip()
    return is_admin_email(email)
