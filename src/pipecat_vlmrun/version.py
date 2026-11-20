#
# Copyright (c) 2026, VLM Run
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Package version and the client name sent to the gateway."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pipecat-vlmrun")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"

CLIENT_HEADER = "X-VLMRun-Client"


def client_name() -> str:
    """The value of the ``X-VLMRun-Client`` header, which tells the gateway who is calling."""
    return f"pipecat-vlmrun/{__version__}"
