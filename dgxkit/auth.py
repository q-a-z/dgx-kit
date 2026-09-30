"""Admin password for the dashboard.

The installer stores `salt:scrypt(password)` (hex) in <state>/admin.pw. Signing in
sets a signed session cookie; the signing key lives in <state>/session.key, so
sessions survive a restart and changing the password rotates the key, which
signs everyone out. With no admin.pw (a dev checkout) the dashboard is open.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

COOKIE = "dgxkit_session"
SESSION_SECONDS = 7 * 24 * 3600
MAX_FAILURES, FAILURE_WINDOW = 5, 300  # per client address


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    return salt.hex() + ":" + hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def check_password(password: str, stored: str) -> bool:
    try:
        salt_hex, want = stored.strip().split(":")
        got = hash_password(password, bytes.fromhex(salt_hex)).split(":")[1]
    except ValueError:
        return False
    return hmac.compare_digest(got, want)


def _write_private(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    tmp.replace(path)


class Auth:
    def __init__(self, state_dir: str):
        self.dir = Path(state_dir)
        self.pw_file = self.dir / "admin.pw"
        self.key_file = self.dir / "session.key"
        self.failures: dict[str, deque] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self.pw_file.exists()

    def _key(self) -> bytes:
        if not self.key_file.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            _write_private(self.key_file, secrets.token_hex(32))
        return bytes.fromhex(self.key_file.read_text().strip())

    def _sign(self, expires: int) -> str:
        return hmac.new(self._key(), str(expires).encode(), hashlib.sha256).hexdigest()

    def new_session(self, now: float | None = None) -> str:
        expires = int((now or time.time()) + SESSION_SECONDS)
        return f"{expires}.{self._sign(expires)}"

    def valid(self, token: str | None, now: float | None = None) -> bool:
        if not self.enabled:
            return True
        if not token or "." not in token:
            return False
        exp, sig = token.split(".", 1)
        if not exp.isdigit() or int(exp) < (now or time.time()):
            return False
        return hmac.compare_digest(sig, self._sign(int(exp)))

    def blocked(self, client: str, now: float | None = None) -> bool:
        now = now or time.time()
        q = self.failures[client]
        while q and q[0] < now - FAILURE_WINDOW:
            q.popleft()
        return len(q) >= MAX_FAILURES

    def login(self, password: str, client: str, now: float | None = None) -> str | None:
        """A session token, or None when the password is wrong."""
        if check_password(password, self.pw_file.read_text()):
            self.failures.pop(client, None)
            return self.new_session(now)
        self.failures[client].append(now or time.time())
        return None

    def change(self, old: str, new: str) -> None:
        if self.enabled and not check_password(old, self.pw_file.read_text()):
            raise PermissionError("current password is wrong")
        if len(new) < 8:
            raise ValueError("use at least 8 characters")
        self.dir.mkdir(parents=True, exist_ok=True)
        _write_private(self.pw_file, hash_password(new))
        _write_private(self.key_file, secrets.token_hex(32))  # signs out every other session
