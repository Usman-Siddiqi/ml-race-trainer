import io
import json

import numpy as np
import pygame
import pytest

from race_trainer import cli
from race_trainer.app import MENU_TRACKS, RaceApp
from race_trainer.config import TEST_SEED_BASE, TRACK_NAMES, EnvConfig, TrainConfig
from race_trainer.environment import RaceCarEnv
from race_trainer.rendering import SIZE


def test_default_command_only_opens_menu(monkeypatch):
    seen = []
    monkeypatch.setattr("race_trainer.app.run_app", lambda: seen.append("menu"))
    assert cli.main([]) == 0
    assert seen == ["menu"]


def test_dry_run_does_not_construct_an_agent(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("A dry run must not create a neural network")

    monkeypatch.setattr("race_trainer.agent.DQNAgent.__init__", forbidden)
    monkeypatch.setattr("torch.cuda.is_available", forbidden)
    arguments = ["train", "--dry-run", "--device", "cpu", "--steps", "5000", "--envs", "8"]
    assert cli.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training_started"] is False
    assert result["training"]["total_steps"] == 5000 and result["training"]["num_envs"] == 8
    assert result["device"]["name"] == "cpu"
    # By default the cars train on generated layouts and are scored on unseen ones.
    assert result["training"]["tracks"] == ["random"]
    held_out = set(result["training"]["eval_tracks"])
    assert not held_out & set(TrainConfig().training_specs()) and len(held_out) == 16
    assert cli.main([*arguments, "--tracks", "oval", "random:7", "random"]) == 0
    assert json.loads(capsys.readouterr().out)["training"]["tracks"] == [
        "oval",
        "random:7",
        "random",
    ]


def test_power_aware_device_selection_is_the_training_default(monkeypatch, capsys):
    monkeypatch.setattr("race_trainer.device.ac_power_connected", lambda: False)
    assert TrainConfig().device == "auto"
    assert cli.main(["train", "--dry-run"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["device"]["name"] == "cpu" and "battery" in result["device"]["reason"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--dry-run", "--device", "cpu", "--steps", "0"],
        ["train", "--dry-run", "--device", "cpu", "--batch-size", "999999"],
        ["train", "--dry-run", "--device", "cpu", "--envs", "0"],
        ["train", "--dry-run", "--device", "cpu", "--track-pool", "2000000000"],
        ["evaluate", "--model", "missing.pt", "--episodes", "0"],
        ["evaluate", "--model", "missing.pt"],
        ["watch", "--model", "missing.pt"],
    ],
)
def test_invalid_configuration_fails_before_work(arguments):
    assert cli.main(arguments) == 2


@pytest.mark.parametrize(
    "arguments",
    [["train", "--tracks", "figure8"], ["manual", "--track", "random"], ["watch", "--model"]],
)
def test_unknown_tracks_are_rejected_by_the_parser(arguments):
    with pytest.raises(SystemExit):
        cli.main(arguments)


def test_tracks_command_explains_the_held_out_seed_ranges(capsys):
    assert cli.main(["tracks"]) == 0
    output = capsys.readouterr().out
    assert all(name in output for name in TRACK_NAMES)
    assert "random:0 .. random:511" in output and f"random:{TEST_SEED_BASE}+" in output


@pytest.mark.parametrize("track", ["oval", "s_track", "square", "random:1", "random:3"])
def test_offscreen_frame_has_expected_dimensions(track):
    env = RaceCarEnv(EnvConfig(track=track), render_mode="rgb_array")
    env.reset(seed=7)
    frame = env.render()
    assert frame.shape == (SIZE[1], SIZE[0], 3)
    assert frame.dtype == np.uint8 and frame.std() > 20
    env.close()
    env.close()


def test_menu_setup_and_navigation_never_launch_a_process(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Opening the app or setup must not launch training")

    monkeypatch.setattr("race_trainer.app.subprocess.Popen", forbidden)
    app = RaceApp()
    try:
        app.draw()
        app.click("track")
        assert app.track_name == "s_track"
        for _ in range(3):
            app.click("track")
        # Beyond the built-in layouts the menu offers generated ones that are never trained on.
        assert app.track_name == f"random:{TEST_SEED_BASE + 1}" and "Generated" in app.track_label
        app.draw()
        app.click("setup")
        app.click("device")
        app.click("fewer")
        app.draw()
        assert app.mode == "setup" and app.steps == TrainConfig().total_steps - 250_000
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert app.mode == "menu"
        app.handle_event(pygame.event.Event(pygame.QUIT))
        assert not app.running
    finally:
        pygame.display.quit()
    assert not set(MENU_TRACKS) & set(TrainConfig().training_specs())


def test_monitor_process_lifecycle_with_a_fake_child_only(monkeypatch, tmp_path):
    commands = []

    class FakeChild:
        returncode = None

        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = io.StringIO("")

        def poll(self):
            return self.returncode

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("race_trainer.app.subprocess.Popen", FakeChild)
    app = RaceApp()
    try:
        app.click("setup")
        assert not commands
        app.click("start")
        assert commands[0][1:4] == ["-m", "race_trainer", "train"]
        assert "--json" in commands[0] and "--tracks" not in commands[0]
        assert commands[0][commands[0].index("--steps") + 1] == str(app.steps)
        assert app.busy and app.mode == "monitor"
        app.output.mkdir(parents=True)
        app.events.put(
            json.dumps(
                {
                    "event": "progress",
                    "steps": 1000,
                    "mean_progress": 0.25,
                    "epsilon": 1.0,
                    "loss": None,
                    "steps_per_second": 5000.0,
                }
            )
        )
        app.events.put(json.dumps({"event": "evaluation", "summary": {"completion_rate": 0.5}}))
        app.poll_job()
        app.draw()
        assert list(app.progress) == [0.25] and app.last_evaluation["completion_rate"] == 0.5
        app.handle_event(pygame.event.Event(pygame.QUIT))
        assert app.closing and app.running and (app.output / "stop.request").exists()
        app.process.returncode = 0
        app.poll_job()
        assert not app.running
    finally:
        pygame.display.quit()


def test_viewer_closes_and_steps_to_unseen_tracks_without_training(monkeypatch):
    from race_trainer import viewer

    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    assert viewer.play(EnvConfig()) is True
    seen = []
    events = [
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events.pop(0))
    draw = viewer.RaceRenderer.draw

    def record(self, env, **kwargs):
        seen.append(env.track.name)
        return draw(self, env, **kwargs)

    monkeypatch.setattr(viewer.RaceRenderer, "draw", record)
    assert viewer.play(EnvConfig()) is False
    assert seen == [f"random:{TEST_SEED_BASE}", f"random:{TEST_SEED_BASE + 1}"]
