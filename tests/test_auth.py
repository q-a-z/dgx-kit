import subprocess

from fastapi.testclient import TestClient

from dgxkit.app import create_app
from dgxkit.auth import Auth, check_password, hash_password


def test_installer_hash_format_is_accepted():
    # Same one-liner install.sh uses to store the admin password.
    stored = subprocess.run(
        ["python3", "-c", "import hashlib,os,sys; s=os.urandom(16); print(s.hex()+':'+hashlib.scrypt("
         "sys.stdin.buffer.read(),salt=s,n=2**14,r=8,p=1).hex())"],
        input=b"correct horse", capture_output=True, check=True).stdout.decode()
    assert check_password("correct horse", stored)
    assert not check_password("wrong", stored)
    assert not check_password("x", "garbage")


def test_sessions_expire_and_die_with_a_password_change(tmp_path):
    a = Auth(str(tmp_path))
    assert a.valid(None)  # no password set: open
    (tmp_path / "admin.pw").write_text(hash_password("first-pass"))
    t = a.login("first-pass", "1.2.3.4", now=1000)
    assert a.valid(t, now=2000) and not a.valid(t, now=1000 + 8 * 86400)
    tampered = t[:-1] + ("1" if t[-1] == "0" else "0")
    assert not a.valid(tampered, now=2000) and not a.valid("123.abc", now=0)
    a.change("first-pass", "second-pass")
    assert not a.valid(t, now=2000)


def test_too_many_wrong_passwords_blocks_the_client(tmp_path):
    a = Auth(str(tmp_path))
    (tmp_path / "admin.pw").write_text(hash_password("right-pass"))
    for i in range(5):
        assert a.login("nope", "9.9.9.9", now=100 + i) is None
    assert a.blocked("9.9.9.9", now=110) and not a.blocked("8.8.8.8", now=110)
    assert not a.blocked("9.9.9.9", now=500)


def test_api_needs_a_session_once_a_password_exists(env):
    client, s, _ = env
    (tmp := s.state_dir)
    from pathlib import Path
    Path(tmp).mkdir(parents=True, exist_ok=True)
    (Path(tmp) / "admin.pw").write_text(hash_password("letmein-1"))
    assert client.get("/api/models").status_code == 401
    me = client.get("/api/me").json()
    assert me.pop("version")  # the running version is public: the sign-in card shows it too
    assert me == {"auth_required": True, "signed_in": False, "readonly": False}
    assert client.post("/api/login", json={"password": "bad"}).status_code == 401
    assert client.post("/api/login", json={"password": "letmein-1"}).status_code == 200
    assert client.get("/api/models").status_code == 200
    assert client.post("/api/password", json={"old": "bad", "new": "whatever-2"}).status_code == 403
    assert client.post("/api/password", json={"old": "letmein-1", "new": "short"}).status_code == 422
    assert client.post("/api/password", json={"old": "letmein-1", "new": "letmein-2"}).status_code == 200
    assert client.get("/api/models").status_code == 200  # the changer stays signed in
    client.post("/api/logout")
    assert client.get("/api/models").status_code == 401
