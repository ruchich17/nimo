"""Spoken plan via macOS `say` (offline). Non-blocking; a new line cuts off the old one."""
import shutil
import subprocess


class Voice:
    def __init__(self, enabled=True, rate=190):
        self.enabled = enabled and shutil.which("say") is not None
        self.rate = rate
        self.proc = None

    def say(self, text):
        if not self.enabled or not text:
            return
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = subprocess.Popen(["say", "-r", str(self.rate), text])
