"""Barebones Panel dashboard package."""

__version__ = "0.1.0"

from .app import create_app, serve_app

__all__ = ["create_app", "serve_app"]
