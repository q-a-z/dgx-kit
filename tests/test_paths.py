from dgxkit.paths import expand_home


def test_tilde_is_the_persons_home_even_when_the_service_runs_as_someone_else(tmp_path, monkeypatch):
    homes = tmp_path / "home"
    (homes / "alice" / "llm-ctl" / "models").mkdir(parents=True)
    (homes / "other").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "root"))  # what ~ is inside the installer's container
    monkeypatch.delenv("DGXKIT_HOME", raising=False)
    want = str(homes / "alice" / "llm-ctl" / "models")
    assert expand_home("~/llm-ctl/models", homes=str(homes)) == want  # the one home that has it
    monkeypatch.setenv("DGXKIT_HOME", str(homes / "alice"))
    assert expand_home("~/llm-ctl/models", homes=str(homes)) == want  # the installer's record wins
    assert expand_home("/abs/path") == "/abs/path"
    assert expand_home("~/nowhere/at/all", homes=str(homes)).endswith("alice/nowhere/at/all")  # nothing found: keep the guess


def test_two_homes_with_the_folder_is_not_guessed(tmp_path, monkeypatch):
    homes = tmp_path / "home"
    for u in ("a", "b"):
        (homes / u / "models").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path / "root"))
    monkeypatch.delenv("DGXKIT_HOME", raising=False)
    assert expand_home("~/models", homes=str(homes)) == str(tmp_path / "root" / "models")  # ambiguous: don't pick one
