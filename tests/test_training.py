"""The headless loop on CPU with learning switched off: collection, logs, stop, resume."""

import json
from dataclasses import replace

import numpy as np
import pytest

from race_trainer import cli
from race_trainer.agent import ACTIONS
from race_trainer.checkpoint import load_checkpoint
from race_trainer.config import AgentConfig, EnvConfig, TrainConfig
from race_trainer.environment import VectorRaceEnv
from race_trainer.tracks import make_track
from race_trainer.training import run_training

# Warm-up outlasts every run here, so no gradient step is ever taken on the CPU suite.
AGENT = AgentConfig(hidden_size=8, batch_size=4, replay_capacity=4000, learning_starts=10**9)
ENV = EnvConfig(max_steps=60)
TRAIN = TrainConfig(
    total_steps=400,
    num_envs=4,
    tracks=("random", "oval"),
    track_pool=3,
    eval_tracks=("square", "random:1000000000"),
    eval_every=200,
    log_every=100,
    device="cpu",
)


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_vector_environment_restarts_cars_on_fresh_tracks_and_keeps_true_successors():
    names = ("random:0", "random:1", "oval")
    env = VectorRaceEnv(EnvConfig(max_steps=40), names, 6, seed=3)
    observations = env.reset()
    assert observations.shape == (6, 18) and observations.dtype == np.float32
    seen, ended_total = set(), 0
    for _ in range(120):
        before = [track.name for track in env.sim.geometry.tracks]
        steps = env.sim.steps.copy()
        acting, rewards, terminated, truncated, successors, episodes = env.step(
            np.tile(ACTIONS[2], (6, 1))
        )
        ended = terminated | truncated
        seen.update(before)
        assert rewards.shape == (6,) and not (terminated & truncated).any()
        # Cars that continue act on their successor; restarted cars see a new start.
        np.testing.assert_array_equal(acting[~ended], successors[~ended])
        assert (env.sim.steps[ended] == 0).all() and not env.sim.finished.any()
        assert (env.sim.speed[ended] == 0).all()
        if ended.any():
            ended_total += ended.sum()
            assert len(episodes["reward"]) == ended.sum()
            np.testing.assert_array_equal(episodes["steps"], steps[ended] + 1)
            assert (acting[ended] != successors[ended]).any()
            assert ((episodes["progress"] >= 0) & (episodes["progress"] <= 1)).all()
        else:
            assert episodes is None
    assert seen == set(names) and ended_total > 6
    assert all(env.sim.geometry.on_road(env.sim.corners()).all(axis=1))


def test_headless_run_logs_evaluates_and_saves_without_learning(tmp_path, capsys):
    result = run_training(TRAIN, ENV, AGENT, output=tmp_path / "run")
    assert result["event"] == "finished" and result["steps"] == 400
    text = capsys.readouterr().out
    assert "Training on cpu" in text and "Unseen tracks at 400 steps" in text
    assert "[100.0%]" in text and "warm-up" in text
    config = json.loads((tmp_path / "run" / "config.json").read_text(encoding="utf-8"))
    assert config["resolved_device"] == "cpu" and config["training"]["tracks"] == ["random", "oval"]
    events = read_events(tmp_path / "run" / "metrics.jsonl")
    kinds = [event["event"] for event in events]
    assert kinds[0] == "started" and kinds[-1] == "finished" and kinds.count("progress") == 4
    assert events[0]["training_tracks"] == 4
    progress = [event for event in events if event["event"] == "progress"]
    assert [event["steps"] for event in progress] == [100, 200, 300, 400]
    assert all(event["loss"] is None and event["updates"] == 0 for event in progress)
    assert progress[-1]["steps_per_second"] > 0 and progress[-1]["epsilon"] == 1.0
    evaluations = read_events(tmp_path / "run" / "evaluations.jsonl")
    assert [event["steps"] for event in evaluations] == [200, 400]
    assert set(evaluations[0]["tracks"]) == {"square", "random:1000000000"}
    payload = load_checkpoint(tmp_path / "run" / "latest.pt")
    assert payload["completed_steps"] == 400 and payload["agent"]["updates"] == 0
    assert payload["agent"]["replay"]["size"] == 100
    assert (tmp_path / "run" / "best.pt").exists() and result["best"]
    with pytest.raises(FileExistsError):
        run_training(TRAIN, ENV, AGENT, output=tmp_path / "run")


def test_stop_request_saves_and_resume_continues_in_a_new_directory(tmp_path, monkeypatch):
    first, resumed = tmp_path / "first", tmp_path / "resumed"
    step = VectorRaceEnv.step

    def stop_after_a_while(env, actions):
        if env.sim.steps.max() == 30:
            (first / "stop.request").touch()
        return step(env, actions)

    monkeypatch.setattr(VectorRaceEnv, "step", stop_after_a_while)
    result = run_training(replace(TRAIN, total_steps=4000), ENV, AGENT, output=first)
    monkeypatch.setattr(VectorRaceEnv, "step", step)
    assert result["event"] == "stopped" and 120 <= result["steps"] < 400
    payload = load_checkpoint(first / "latest.pt")
    assert payload["completed_steps"] == result["steps"]
    assert read_events(first / "metrics.jsonl")[-1]["event"] == "stopped"
    # Resume keeps every saved setting; only the target, device, and threads may change.
    with pytest.raises(ValueError, match="preserve"):
        run_training(replace(TRAIN, seed=1), ENV, AGENT, output=resumed, resume=first / "latest.pt")
    with pytest.raises(ValueError, match="preserve"):
        run_training(
            TRAIN, replace(ENV, grip=200.0), AGENT, output=resumed, resume=first / "latest.pt"
        )
    with pytest.raises(ValueError, match="total target"):
        run_training(
            replace(TRAIN, total_steps=100), ENV, AGENT, output=resumed, resume=first / "latest.pt"
        )
    final = run_training(
        replace(TRAIN, total_steps=600), ENV, AGENT, output=resumed, resume=first / "latest.pt"
    )
    assert final["event"] == "finished" and final["steps"] == 600
    # The resumed directory is self-contained even if it never beats the earlier best.
    assert (resumed / "best.pt").exists() and final["best"]
    assert final["steps_this_run"] == 600 - result["steps"]
    events = read_events(resumed / "metrics.jsonl")
    assert events[0]["completed_steps"] == result["steps"]
    restored = load_checkpoint(resumed / "latest.pt")
    replay = restored["agent"]["replay"]
    assert replay["size"] == 150
    # Cars restart on resume, so the last step before the stop must end its n-step chain.
    boundary = result["steps"] // 4 - 1
    assert replay["ended"][boundary].all()
    config = json.loads((resumed / "config.json").read_text(encoding="utf-8"))
    assert config["resumed_from"] == str(first / "latest.pt")


def test_cli_trains_evaluates_on_unseen_tracks_and_reports_what_was_seen(tmp_path, capsys):
    run = tmp_path / "run"
    arguments = ["train", "--device", "cpu", "--steps", "240", "--envs", "4", "--max-steps", "40"]
    arguments += ["--tracks", "oval", "random", "--track-pool", "2", "--learning-starts", "999999"]
    arguments += ["--eval-every", "240", "--log-every", "120", "--output", str(run), "--json"]
    assert cli.main(arguments) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["event"] for line in lines][-1] == "finished"
    report = tmp_path / "report.json"
    evaluate = ["evaluate", "--model", str(run / "best.pt"), "--generated", "2"]
    assert cli.main([*evaluate, "--output", str(report)]) == 0
    table = capsys.readouterr().out
    assert "UNSEEN (4 tracks)" in table and "random:2000000001" in table
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["seen_in_training"] == ["oval"] and saved["summary"]["episodes"] == 5
    assert cli.main([*evaluate, "--tracks", "random:0", "random:9", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["seen_in_training"] == ["random:0"] and set(shown["tracks"]) == {
        "random:0",
        "random:9",
    }
    assert make_track("random:9").name == "random:9"


def test_simulator_benchmark_is_bounded_and_rejects_long_runs(capsys):
    from race_trainer.benchmark import benchmark_simulator

    assert benchmark_simulator(4, seconds=0.05) > 0
    assert cli.main(["benchmark", "--seconds", "0"]) == 2
    assert cli.main(["benchmark", "--seconds", "60"]) == 2
    assert "between 0 and 10" in capsys.readouterr().err
