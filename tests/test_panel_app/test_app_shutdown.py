"""Tests for panel_app.app startup guard and shutdown helpers."""

import os
import signal
import socket
from unittest.mock import MagicMock, patch

import pytest


class TestCheckPortAvailable:
    """_check_port_available should fail fast when port is occupied."""

    def test_succeeds_on_free_port(self):
        from panel_app.app import _check_port_available

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _, free_port = sock.getsockname()
        sock.close()

        _check_port_available("localhost", free_port)

    def test_raises_system_exit_on_occupied_port(self):
        from panel_app.app import _check_port_available

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _, occupied_port = sock.getsockname()
        sock.listen(1)
        try:
            with pytest.raises(SystemExit):
                _check_port_available("localhost", occupied_port)
        finally:
            sock.close()

    def test_resolves_localhost_to_loopback(self):
        """'localhost' should probe 127.0.0.1, not a DNS-resolved address."""
        from panel_app.app import _check_port_available

        with patch("panel_app.app.socket") as mock_socket_mod:
            mock_sock = MagicMock()
            mock_socket_mod.socket.return_value = mock_sock
            mock_socket_mod.AF_INET = socket.AF_INET
            mock_socket_mod.SOCK_STREAM = socket.SOCK_STREAM

            _check_port_available("localhost", 9999)

            mock_sock.bind.assert_called_once_with(("127.0.0.1", 9999))
            mock_sock.close.assert_called_once()


class TestRequestShutdown:
    """_request_shutdown should prefer graceful IO-loop stop."""

    def _reset_module_state(self, app_mod):
        app_mod._active_server = None
        app_mod._shutdown_flag = False

    def test_graceful_ioloop_stop(self):
        import panel_app.app as app_mod

        mock_loop = MagicMock()
        mock_server = MagicMock()
        mock_server.io_loop = mock_loop

        app_mod._active_server = mock_server
        app_mod._shutdown_flag = False
        try:
            app_mod._request_shutdown()
            mock_loop.add_callback.assert_called_once_with(mock_loop.stop)
            assert app_mod._shutdown_flag is True
        finally:
            self._reset_module_state(app_mod)

    def test_idempotent_when_already_shutting_down(self):
        import panel_app.app as app_mod

        mock_server = MagicMock()
        mock_server.io_loop = MagicMock()
        app_mod._active_server = mock_server
        app_mod._shutdown_flag = True
        try:
            app_mod._request_shutdown()
            mock_server.io_loop.add_callback.assert_not_called()
        finally:
            self._reset_module_state(app_mod)

    def test_sigterm_fallback_when_no_server(self):
        import panel_app.app as app_mod

        app_mod._active_server = None
        app_mod._shutdown_flag = False
        try:
            with patch("panel_app.app.os.kill") as mock_kill:
                app_mod._request_shutdown()
                mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
                assert app_mod._shutdown_flag is True
        finally:
            self._reset_module_state(app_mod)

    def test_sigterm_fallback_when_server_lacks_ioloop(self):
        import panel_app.app as app_mod

        mock_server = MagicMock(spec=[])  # no attributes at all
        app_mod._active_server = mock_server
        app_mod._shutdown_flag = False
        try:
            with patch("panel_app.app.os.kill") as mock_kill:
                app_mod._request_shutdown()
                mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
        finally:
            self._reset_module_state(app_mod)
