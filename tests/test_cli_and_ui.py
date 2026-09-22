import io
import json

import numpy as np
import pygame
import pytest

from race_trainer import cli
from race_trainer.app import RaceApp
from race_trainer.config import EnvConfig
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
    assert cli.main(["train", "--dry-run", "--episodes", "10"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["training_started"] is False
    assert result["training"]["episodes"] == 10
    assert result["device"]["name"] == "cpu"
    assert result["environment"]["track"] == result["training"]["tracks"][0]


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--dry-run", "--episodes", "0"],
        ["train", "--dry-run", "--batch-size", "999999"],
        ["train", "--dry-run", "--render-every", "-1"],
        ["evaluate", "--model", "missing.pt", "--episodes", "0"],
    ],
)
def test_invalid_configuration_fails_before_work(arguments):
    assert cli.main(arguments) == 2


@pytest.mark.parametrize("track", ["oval", "s_track", "square"])
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
        app.click("setup")
        app.click("device")
        app.click("fewer")
        app.draw()
        assert app.mode == "setup" and app.episodes == 900
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        assert app.mode == "menu"
        app.handle_event(pygame.event.Event(pygame.QUIT))
        assert not app.running
    finally:
        pygame.display.quit()


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
        assert app.busy and app.mode == "monitor"
        app.output.mkdir(parents=True)
        app.events.put(
            json.dumps({"event": "episode", "episode": 1, "reward": 2.0, "epsilon": 1.0})
        )
        app.poll_job()
        app.draw()
        assert list(app.rewards) == [2.0]
        app.handle_event(pygame.event.Event(pygame.QUIT))
        assert app.closing and app.running and (app.output / "stop.request").exists()
        app.process.returncode = 0
        app.poll_job()
        assert not app.running
    finally:
        pygame.display.quit()


def test_manual_viewer_handles_close_without_training(monkeypatch):
    from race_trainer.viewer import play

    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    assert play(EnvConfig()) is True
