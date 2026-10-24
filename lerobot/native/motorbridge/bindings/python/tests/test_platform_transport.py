from __future__ import annotations

import argparse
import importlib
import io
import subprocess
from unittest import mock

from motorbridge import gateway
cli_main = importlib.import_module("motorbridge.cli.main")
from motorbridge.platform_hints import (
    effective_transport_for_preflight,
    should_skip_runtime_preflight,
    with_inferred_dm_serial_transport,
)


def test_runtime_preflight_skips_help_and_version_flags() -> None:
    assert should_skip_runtime_preflight(["--help"])
    assert should_skip_runtime_preflight(["--version"])
    assert should_skip_runtime_preflight(["-v"])


def test_damiao_serial_port_infers_dm_serial_transport() -> None:
    args = ["--vendor", "damiao", "--serial-port", "COM7"]

    assert effective_transport_for_preflight(args) == "dm-serial"
    assert with_inferred_dm_serial_transport(args) == [
        "--transport",
        "dm-serial",
        "--vendor",
        "damiao",
        "--serial-port",
        "COM7",
    ]


def test_non_damiao_serial_port_does_not_infer_dm_serial() -> None:
    args = ["--vendor", "robstride", "--serial-port", "COM7"]

    assert effective_transport_for_preflight(args) == "auto"
    assert with_inferred_dm_serial_transport(args) == args


def test_explicit_auto_transport_is_rewritten_for_damiao_serial_port() -> None:
    args = ["--transport", "auto", "--vendor", "damiao", "--serial-port", "COM7"]

    assert with_inferred_dm_serial_transport(args) == [
        "--transport",
        "dm-serial",
        "--vendor",
        "damiao",
        "--serial-port",
        "COM7",
    ]


def test_gateway_version_does_not_run_runtime_preflight() -> None:
    with (
        mock.patch.object(gateway, "_resolve_gateway_binary", return_value="ws_gateway"),
        mock.patch.object(gateway, "preflight_can_runtime") as preflight,
        mock.patch.object(subprocess, "call", return_value=0) as call,
    ):
        assert gateway.run_gateway(["--version"]) == 0

    preflight.assert_not_called()
    call.assert_called_once_with(["ws_gateway", "--version"])


def test_gateway_without_args_prints_gateway_help() -> None:
    with (
        mock.patch.object(gateway, "_resolve_gateway_binary", return_value="ws_gateway"),
        mock.patch.object(gateway, "preflight_can_runtime") as preflight,
        mock.patch.object(subprocess, "call", return_value=0) as call,
    ):
        assert gateway.run_gateway([]) == 0

    preflight.assert_not_called()
    call.assert_called_once_with(["ws_gateway", "--help"])


def test_gateway_normalizes_damiao_serial_args_before_launch() -> None:
    with (
        mock.patch.object(gateway, "_resolve_gateway_binary", return_value="ws_gateway"),
        mock.patch.object(gateway, "preflight_can_runtime", return_value=None) as preflight,
        mock.patch.object(subprocess, "call", return_value=0) as call,
    ):
        assert gateway.run_gateway(["--vendor", "damiao", "--serial-port", "COM7"]) == 0

    preflight.assert_called_once_with("motorbridge-gateway", "dm-serial", "can0")
    call.assert_called_once_with(
        [
            "ws_gateway",
            "--transport",
            "dm-serial",
            "--vendor",
            "damiao",
            "--serial-port",
            "COM7",
        ]
    )


def test_cli_normalizes_damiao_serial_args_before_opening_controller() -> None:
    args = argparse.Namespace(
        command="run",
        vendor="damiao",
        transport="auto",
        channel="can0",
        serial_port="COM7",
        _provided_options={"serial-port"},
    )

    with (
        mock.patch.object(cli_main, "_parse_with_legacy_support", return_value=args),
        mock.patch.object(cli_main, "preflight_can_runtime", return_value=None) as preflight,
        mock.patch.object(cli_main, "_run_command") as run_command,
    ):
        cli_main.main()

    assert args.transport == "dm-serial"
    preflight.assert_called_once_with("motorbridge-cli", "dm-serial", "can0")
    run_command.assert_called_once_with(args)


def test_cli_without_args_prints_help_without_running_command() -> None:
    out = io.StringIO()
    with (
        mock.patch.object(cli_main.sys, "argv", ["motorbridge-cli"]),
        mock.patch.object(cli_main.sys, "stdout", out),
        mock.patch.object(cli_main, "preflight_can_runtime") as preflight,
        mock.patch.object(cli_main, "_run_command") as run_command,
    ):
        try:
            cli_main.main()
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("expected SystemExit")

    text = out.getvalue()
    assert "motorbridge Python SDK CLI" in text
    assert "{run,id-dump,id-set,scan" in text
    preflight.assert_not_called()
    run_command.assert_not_called()
