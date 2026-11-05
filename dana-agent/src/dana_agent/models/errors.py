"""Errors raised by the model registry (mapped to HTTP codes by the routes)."""

from __future__ import annotations


class ModelsError(Exception):
    """Base class for registry errors."""


class HubAuthError(ModelsError):
    """The hub refused access (401/403): gated model or bad/missing token."""


class HubUnavailableError(ModelsError):
    """The hub could not be reached or returned a server-side error."""


class ModelNotFoundError(ModelsError):
    """Unknown repo / job / file."""


class ModelBusyError(ModelsError):
    """The operation conflicts with an in-flight pull."""


class ModelInputError(ModelsError):
    """Bad caller input (invalid repo id, unsafe path, unloadable model, ...)."""


GATED_MESSAGE = "gated model: accept the license on huggingface.co and set HF_TOKEN"
