"""All verification is CPU only. No test may train a neural network."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
