# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for forge.actors.coder.SandboxedPythonCoder.
"""
import os
import uuid
from contextlib import asynccontextmanager
from unittest.mock import Mock, patch

import pytest
from forge.actors.coder import SandboxedPythonCoder

from monarch.actor import this_proc


@asynccontextmanager
async def create_mock_coder(
    execute_stdout="hello world\n",
    execute_returncode=0,
    execute_stderr="",
    import_fails=False,
    create_fails=False,
):
    """Context manager that creates a mocked SandboxedPythonCoder."""
    unique_id = str(uuid.uuid4())[:8]
    container_name = f"test_sandbox_{unique_id}"

    image_path = f"/tmp/test_mock_image_{unique_id}.sqsh"

    coder = None
    try:
        with patch("subprocess.run") as mock_run:

            def mock_subprocess_run(*args, **kwargs):
                cmd = args[0]
                if "import" in cmd:
                    result = Mock()
                    if import_fails:
                        result.returncode = 1
                        result.stderr = "Failed to import image: network error"
                    else:
                        result.returncode = 0
                        result.stderr = ""
                    return result
                elif "remove" in cmd:
                    result = Mock()
                    result.returncode = 0
                    return result
                elif "create" in cmd:
                    result = Mock()
                    if create_fails:
                        result.returncode = 1
                        result.stderr = "Failed to create container: no space"
                    else:
                        result.returncode = 0
                        result.stderr = ""
                    return result
                elif "start" in cmd:
                    result = Mock()
                    result.returncode = execute_returncode
                    result.stdout = execute_stdout
                    result.stderr = execute_stderr
                    return result
                else:
                    raise ValueError(f"Unexpected subprocess call: {cmd}")

            mock_run.side_effect = mock_subprocess_run

            coder = this_proc().spawn(
                f"coder_{uuid.uuid1()}",
                SandboxedPythonCoder,
                "docker://python:3.10",
                image_path,
                container_name,
            )

            yield coder, mock_run

    finally:
        if coder:
            await SandboxedPythonCoder.shutdown(coder)

        if os.path.exists(image_path):
            os.unlink(image_path)


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_success():
    """Test successful execution."""
    async with create_mock_coder(execute_stdout="Hello World\n") as (coder, _):
        await coder.setup.call_one()
        result, _ = await coder.execute.call_one(code="print('Hello World')")
        assert result == "Hello World\n"


@pytest.mark.timeout(10)
@pytest.mark.asyncio
async def test_coder_execution_failure():
    """Test execution failure."""
    async with create_mock_coder(
        execute_returncode=1, execute_stderr="SyntaxError: invalid syntax"
    ) as (coder, _):
        await coder.setup.call_one()
        output, err = await coder.execute.call_one(code="invalid syntax")
        assert "SyntaxError" in err
