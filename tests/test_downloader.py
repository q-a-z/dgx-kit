from pathlib import Path

from dgxkit.downloader import Job, dir_bytes, fits_on_disk


def test_fits_on_disk_counts_what_is_already_there(tmp_path):
    assert fits_on_disk(str(tmp_path), needed=10, already=0, reserve=0)
    assert not fits_on_disk(str(tmp_path), needed=10**18, already=0, reserve=0)
    assert fits_on_disk(str(tmp_path), needed=10**18, already=10**18, reserve=0)


def test_progress_reports_bytes_and_pct(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"x" * 250)
    job = Job("org/M", total_bytes=1000, dest=Path(tmp_path))
    p = job.progress()
    assert p["bytes"] == dir_bytes(tmp_path) == 250 and p["pct"] == 25.0


def test_pause_keeps_the_job_and_resume_restarts_it(tmp_path):
    import asyncio
    from dgxkit.downloader import Downloader, Job

    class Proc:
        killed = False

        def kill(self):
            self.killed = True

    d = Downloader(str(tmp_path))
    runs = []

    async def run(job):
        runs.append(job.revision)

    d._run = run
    job = Job("org/m", 10, tmp_path / "m", None, "abc", state="running", proc=Proc())
    d.jobs["org/m"] = job
    assert d.resume("org/m") is False
    assert d.pause("org/m") and job.state == "paused" and job.proc.killed
    assert d.pause("org/m") is False

    async def go():
        assert d.resume("org/m")
        await asyncio.sleep(0)

    asyncio.run(go())
    assert job.state == "queued" and runs == ["abc"]
    job.state = "paused"
    d.cancel("org/m")
    assert job.state == "cancelled"
