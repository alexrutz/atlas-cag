"""Running short helper commands (nvidia-smi, nvcc, llama-server --version) without ever hanging.

subprocess.run(timeout=…) kills a command that times out and then waits for it to exit. A command
stuck in the GPU driver (nvidia-smi under WSL when the driver hangs) cannot exit, so that wait never
returns and blocks the caller for good. run() gives up on such a command and lets a background
thread collect it whenever it finally exits.
"""

import subprocess
import threading


def run(cmd: list[str], timeout: float, env: dict | None = None) -> subprocess.CompletedProcess | None:
    """Run a command and capture its output; None if it cannot start or does not finish in time."""
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                                start_new_session=True)
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        threading.Thread(target=proc.wait, daemon=True).start()  # reaped when it exits, if ever
        return None
    return subprocess.CompletedProcess(cmd, proc.returncode, out, "")
