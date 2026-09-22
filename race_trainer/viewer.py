"""Manual driving and policy replay share the same event loop and simulator."""

import pygame

from .environment import RaceCarEnv
from .rendering import RaceRenderer


def play(env_config, policy=None, *, seed=42, episodes=1):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = RaceCarEnv(env_config)
    renderer = RaceRenderer("human")
    observation, _ = env.reset(seed=seed)
    paused, running, episode = False, True, 0
    window_closed = False
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    running = False
                    window_closed = event.type == pygame.QUIT
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_TAB:
                        renderer.show_sensors = not renderer.show_sensors
                    elif event.key == pygame.K_r:
                        observation, _ = env.reset(seed=seed + episode)
                        paused = False
                    elif event.key == pygame.K_RETURN and env._finished and policy is not None:
                        episode += 1
                        if episode >= episodes:
                            running = False
                        else:
                            observation, _ = env.reset(seed=seed + episode)
            if not running:
                break
            if not paused and not env._finished:
                if policy is None:
                    keys = pygame.key.get_pressed()
                    throttle = int(keys[pygame.K_w] or keys[pygame.K_UP]) - int(
                        keys[pygame.K_s] or keys[pygame.K_DOWN]
                    )
                    steering = int(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - int(
                        keys[pygame.K_a] or keys[pygame.K_LEFT]
                    )
                    action = (throttle, steering)
                else:
                    from .agent import ACTIONS

                    action = ACTIONS[policy.act(observation, explore=False)]
                observation, _, _, _, _ = env.step(action)
            renderer.draw(
                env, mode_label="MANUAL DRIVE" if policy is None else "MODEL REPLAY", paused=paused
            )
            if policy is not None and env._finished:
                from .rendering import MUTED, text

                text(renderer.surface, "ENTER  Next episode / finish replay", (25, 94), 13, MUTED)
                pygame.display.flip()
    finally:
        env.close()
        renderer.close()
    return window_closed
