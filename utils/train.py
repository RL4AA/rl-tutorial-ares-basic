# Slightly modified version of ea_train.py to make it run in the workshop context.


import pathlib
import pickle
from functools import partial

import cheetah
import cv2
import gym
import numpy as np
import yaml
from gym import spaces
from gym.wrappers import (
    FilterObservation,
    FlattenObservation,
    FrameStack,
    RecordVideo,
    RescaleAction,
    TimeLimit,
)
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .utils import FilterAction


def main():
    config = {
        "action_mode": "direct",
        "gamma": 0.99,
        "filter_action": [0, 1, 3],
        "filter_observation": None,
        "frame_stack": None,
        "incoming_mode": "random",
        "incoming_values": None,
        "magnet_init_mode": "constant",
        "magnet_init_values": np.array([10, -10, 0, 10, 0]),
        "misalignment_mode": "constant",
        "misalignment_values": np.zeros(8),
        "n_envs": 40,
        "normalize_observation": True,
        "normalize_reward": True,
        "rescale_action": (-3, 3),
        "reward_mode": "feedback",
        "sb3_device": "auto",
        "target_beam_mode": "constant",
        "target_beam_values": np.zeros(4),
        "target_mu_x_threshold": np.inf,
        "target_mu_y_threshold": np.inf,
        "target_sigma_x_threshold": 1e-4,
        "target_sigma_y_threshold": 1e-4,
        "threshold_hold": 5,
        "time_limit": 25,
        "vec_env": "subproc",
    }

    train(config)


def train(config):
    print(f"==> Training agent \"{config['run_name']}\"")

    config["wandb_run_name"] = config["run_name"]
    run_name = config["run_name"]

    # Fill in defaults for MLE School
    config["vec_env"] = "subproc"
    config["incoming_mode"] = "random"
    config["incoming_values"] = None
    config["sb3_device"] = "auto"
    config["filter_action"] = [0, 1, 3]
    config["filter_observation"] = ["beam", "magnets"]
    config["magnet_init_mode"] = "constant"
    config["magnet_init_values"] = np.zeros(5)
    config["misalignment_mode"] = "constant"
    config["misalignment_values"] = np.zeros(8)
    config["target_beam_mode"] = "constant"
    config["target_beam_values"] = np.zeros(4)
    config["target_mu_x_threshold"] = np.inf
    config["target_mu_y_threshold"] = np.inf
    config["vec_env"] = "dummy"

    # Setup environments
    if config["vec_env"] == "dummy":
        env = DummyVecEnv([partial(make_env, config) for _ in range(config["n_envs"])])
    elif config["vec_env"] == "subproc":
        env = SubprocVecEnv(
            [partial(make_env, config) for _ in range(config["n_envs"])]
        )
    else:
        raise ValueError(f"Invalid value \"{config['vec_env']}\" for dummy")
    pathlib.Path(f"utils/monitors/{config['run_name']}").mkdir(
        parents=True, exist_ok=True
    )
    eval_env = DummyVecEnv(
        [
            partial(
                make_env,
                config,
                record_video=True,
                monitor_filename=f"utils/monitors/{config['run_name']}/0",
            )
        ]
    )

    if config["normalize_observation"] or config["normalize_reward"]:
        env = VecNormalize(
            env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=config["normalize_observation"],
            norm_reward=config["normalize_reward"],
            gamma=config["gamma"],
            training=False,
        )

    # Train
    model = PPO(
        "MlpPolicy",
        env,
        device=config["sb3_device"],
        gamma=config["gamma"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        policy_kwargs={"net_arch": config["net_arch"]},
        verbose=0,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        eval_env=eval_env,
        eval_freq=500,
    )

    model.save(f"utils/models/{run_name}/model")
    if config["normalize_observation"] or config["normalize_reward"]:
        env.save(f"utils/models/{run_name}/normalizer")
    save_to_yaml(config, f"utils/models/{run_name}/config")


def make_env(config, record_video=False, monitor_filename=None):
    env = ARESEACheetah(
        incoming_mode=config["incoming_mode"],
        incoming_values=config["incoming_values"],
        misalignment_mode=config["misalignment_mode"],
        misalignment_values=config["misalignment_values"],
        abort_if_off_screen=config["abort_if_off_screen"],
        action_mode=config["action_mode"],
        magnet_init_mode=config["magnet_init_mode"],
        magnet_init_values=config["magnet_init_values"],
        reward_mode=config["reward_mode"],
        target_beam_mode=config["target_beam_mode"],
        target_beam_values=config["target_beam_values"],
        target_mu_x_threshold=config["target_mu_x_threshold"],
        target_mu_y_threshold=config["target_mu_y_threshold"],
        target_sigma_x_threshold=config["target_sigma_x_threshold"],
        target_sigma_y_threshold=config["target_sigma_y_threshold"],
        threshold_hold=config["threshold_hold"],
        time_reward=config["time_reward"],
    )
    if config["filter_observation"] is not None:
        env = FilterObservation(env, config["filter_observation"])
    if config["filter_action"] is not None:
        env = FilterAction(env, config["filter_action"], replace=0)
    if config["time_limit"] is not None:
        env = TimeLimit(env, config["time_limit"])
    env = FlattenObservation(env)
    if config["frame_stack"] is not None:
        env = FrameStack(env, config["frame_stack"])
    if config["rescale_action"] is not None:
        env = RescaleAction(
            env, config["rescale_action"][0], config["rescale_action"][1]
        )
    env = Monitor(env, filename=monitor_filename)
    if record_video:
        env = RecordVideo(
            env, video_folder=f"utils/recordings/{config['wandb_run_name']}"
        )
    return env


class ARESEA(gym.Env):
    """
    Base class for beam positioning and focusing on AREABSCR1 in the ARES EA.

    Parameters
    ----------
    action_mode : str
        How actions work. Choose `"direct"`, `"direct_unidirectional_quads"` or
        `"delta"`.
    magnet_init_mode : str
        Magnet initialisation on `reset`. Set to `None`, `"random"` or `"constant"`. The
        `"constant"` setting requires `magnet_init_values` to be set.
    magnet_init_values : np.ndarray
        Values to set magnets to on `reset`. May only be set when `magnet_init_mode` is
        set to `"constant"`.
    reward_mode : str
        How to compute the reward. Choose from `"feedback"` or `"differential"`.
    target_beam_mode : str
        Setting of target beam on `reset`. Choose from `"constant"` or `"random"`. The
        `"constant"` setting requires `target_beam_values` to be set.
    """

    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 2}

    def __init__(
        self,
        abort_if_off_screen=False,
        action_mode="delta",
        include_screen_image_in_info=True,
        magnet_init_mode=None,
        magnet_init_values=None,
        reward_mode="negative_objective",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        time_reward=-0.0,
    ):
        self.abort_if_off_screen = abort_if_off_screen
        self.action_mode = action_mode
        self.include_screen_image_in_info = include_screen_image_in_info
        self.magnet_init_mode = magnet_init_mode
        self.magnet_init_values = magnet_init_values
        self.reward_mode = reward_mode
        self.target_beam_mode = target_beam_mode
        self.target_beam_values = target_beam_values
        self.target_mu_x_threshold = target_mu_x_threshold
        self.target_mu_y_threshold = target_mu_y_threshold
        self.target_sigma_x_threshold = target_sigma_x_threshold
        self.target_sigma_y_threshold = target_sigma_y_threshold
        self.threshold_hold = threshold_hold
        self.time_reward = time_reward

        # Create action space
        if self.action_mode == "direct":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )
        elif self.action_mode == "direct_unidirectional_quads":
            self.action_space = spaces.Box(
                low=np.array([0, -72, -6.1782e-3, 0, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 0, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            )
        elif self.action_mode == "delta":
            self.action_space = spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32)
                * 0.1,
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32)
                * 0.1,
            )
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Create observation space
        obs_space_dict = {
            "beam": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            ),
            "magnets": self.action_space
            if self.action_mode.startswith("direct")
            else spaces.Box(
                low=np.array([-72, -72, -6.1782e-3, -72, -6.1782e-3], dtype=np.float32),
                high=np.array([72, 72, 6.1782e-3, 72, 6.1782e-3], dtype=np.float32),
            ),
            "target": spaces.Box(
                low=np.array([-np.inf, 0, -np.inf, 0], dtype=np.float32),
                high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            ),
        }
        obs_space_dict.update(self.get_accelerator_observation_space())
        self.observation_space = spaces.Dict(obs_space_dict)

        # Setup the accelerator (either simulation or the actual machine)
        self.setup_accelerator()

    def reset(self):
        self.reset_accelerator()

        if self.magnet_init_mode == "constant":
            self.set_magnets(self.magnet_init_values)
        elif self.magnet_init_mode == "random":
            self.set_magnets(self.observation_space["magnets"].sample())
        elif self.magnet_init_mode is None:
            pass  # This really is intended to do nothing
        else:
            raise ValueError(
                f'Invalid value "{self.magnet_init_mode}" for magnet_init_mode'
            )

        if self.target_beam_mode == "constant":
            self.target_beam = self.target_beam_values
        elif self.target_beam_mode == "random":
            self.target_beam = self.observation_space["target"].sample()
        else:
            raise ValueError(
                f'Invalid value "{self.target_beam_mode}" for target_beam_mode'
            )

        # Update anything in the accelerator (mainly for running simulations)
        self.update_accelerator()

        self.initial_screen_beam = self.get_beam_parameters()
        self.previous_beam = self.initial_screen_beam
        self.is_in_threshold_history = []
        self.steps_taken = 0

        observation = {
            "beam": self.initial_screen_beam.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32"),
        }
        observation.update(self.get_accelerator_observation())

        return observation

    def step(self, action):
        # Perform action
        if self.action_mode == "direct":
            self.set_magnets(action)
        elif self.action_mode == "direct_unidirectional_quads":
            self.set_magnets(action)
        elif self.action_mode == "delta":
            magnet_values = self.get_magnets()
            self.set_magnets(magnet_values + action)
        else:
            raise ValueError(f'Invalid value "{self.action_mode}" for action_mode')

        # Run the simulation
        self.update_accelerator()

        current_beam = self.get_beam_parameters()
        self.steps_taken += 1

        # Build observation
        observation = {
            "beam": current_beam.astype("float32"),
            "magnets": self.get_magnets().astype("float32"),
            "target": self.target_beam.astype("float32"),
        }
        observation.update(self.get_accelerator_observation())

        # For readibility in computations below
        cb = current_beam
        ib = self.initial_screen_beam
        pb = self.previous_beam
        tb = self.target_beam

        # Compute if done (beam within threshold for a certain time)
        threshold = np.array(
            [
                self.target_mu_x_threshold,
                self.target_sigma_x_threshold,
                self.target_mu_y_threshold,
                self.target_sigma_y_threshold,
            ],
            dtype=np.double,
        )
        threshold = np.nan_to_num(threshold)
        is_in_threshold = np.abs(cb - tb) < threshold
        self.is_in_threshold_history.append(is_in_threshold)
        is_stable_in_threshold = bool(
            np.array(self.is_in_threshold_history[-self.threshold_hold :]).all()
        )
        is_success = is_stable_in_threshold and len(self.is_in_threshold_history) > 5
        is_failure = self.abort_if_off_screen and not self.is_beam_on_screen()
        done = is_success or is_failure

        # Compute reward
        if self.reward_mode == "negative_objective":
            current_objective = np.sum(np.abs(cb - tb)[[1, 3]])
            initial_objective = np.sum(np.abs(ib - tb)[[1, 3]])
            reward = -current_objective / initial_objective
        elif self.reward_mode == "objective_improvement":
            current_objective = np.sum(np.abs(cb - tb)[[1, 3]])
            previous_objective = np.sum(np.abs(pb - tb)[[1, 3]])
            initial_objective = np.sum(np.abs(ib - tb)[[1, 3]])
            reward = (previous_objective - current_objective) / initial_objective
        elif self.reward_mode == "sum_of_pixels":
            screen_image = self.get_screen_image()
            reward = -np.sum(screen_image)
        else:
            raise ValueError(f'Invalid value "{self.reward_mode}" for reward_mode')
        reward += self.time_reward
        reward = float(reward)

        # Put together info
        err_mu_x = abs(cb[0] - tb[0])
        err_sigma_x = abs(cb[1] - tb[1])
        err_mu_y = abs(cb[2] - tb[2])
        err_sigma_y = abs(cb[3] - tb[3])
        mae_focus = (err_sigma_x + err_sigma_y) / 2
        mae_all = (err_mu_x + err_sigma_x + err_mu_y + err_sigma_y) / 4
        info = {
            "binning": self.get_binning(),
            "err_sigma_x": err_sigma_x,
            "err_sigma_y": err_sigma_y,
            "mae_focus": mae_focus,
            "mae_all": mae_all,
            "pixel_size": self.get_pixel_size(),
            "screen_resolution": self.get_screen_resolution(),
        }
        if self.include_screen_image_in_info:
            info["screen_image"] = self.get_screen_image()
        info.update(self.get_accelerator_info())

        self.previous_beam = current_beam

        return observation, reward, done, info

    def render(self, mode="human"):
        assert mode == "rgb_array" or mode == "human"

        binning = self.get_binning()
        pixel_size = self.get_pixel_size()
        resolution = self.get_screen_resolution()

        # Read screen image and make 8-bit RGB
        img = self.get_screen_image()
        img = img / 2**12 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

        # Redraw beam image as if it were binning = 4
        render_resolution = (resolution * binning / 4).astype("int")
        img = cv2.resize(img, render_resolution)

        # Draw desired ellipse
        tb = self.target_beam
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(tb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(tb[1] / pixel_size_b4[0])
        e_pos_y = int(-tb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(tb[3] / pixel_size_b4[1])
        blue = (255, 204, 79)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, blue, 2
        )

        # Draw beam ellipse
        cb = self.get_beam_parameters()
        pixel_size_b4 = pixel_size / binning * 4
        e_pos_x = int(cb[0] / pixel_size_b4[0] + render_resolution[0] / 2)
        e_width_x = int(cb[1] / pixel_size_b4[0])
        e_pos_y = int(-cb[2] / pixel_size_b4[1] + render_resolution[1] / 2)
        e_width_y = int(cb[3] / pixel_size_b4[1])
        red = (0, 0, 255)
        img = cv2.ellipse(
            img, (e_pos_x, e_pos_y), (e_width_x, e_width_y), 0, 0, 360, red, 2
        )

        # Adjust aspect ratio
        new_width = int(img.shape[1] * pixel_size_b4[0] / pixel_size_b4[1])
        img = cv2.resize(img, (new_width, img.shape[0]))

        # Add magnet values and beam parameters
        magnets = self.get_magnets()
        padding = np.full(
            (int(img.shape[0] * 0.27), img.shape[1], 3), fill_value=255, dtype=np.uint8
        )
        img = np.vstack([img, padding])
        black = (0, 0, 0)
        red = (0, 0, 255)
        green = (0, 255, 0)
        img = cv2.putText(
            img, f"Q1={magnets[0]:.2f}", (15, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img, f"Q2={magnets[1]:.2f}", (215, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CV={magnets[2]*1e3:.2f}",
            (415, 545),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        img = cv2.putText(
            img, f"Q3={magnets[3]:.2f}", (615, 545), cv2.FONT_HERSHEY_SIMPLEX, 1, black
        )
        img = cv2.putText(
            img,
            f"CH={magnets[4]*1e3:.2f}",
            (15, 585),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            black,
        )
        mu_x_color = black
        if self.target_mu_x_threshold not in [np.inf, None]:
            mu_x_color = (
                green if abs(cb[0] - tb[0]) < self.target_mu_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"mx={cb[0]*1e3:.2f}",
            (15, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_x_color,
        )
        sigma_x_color = black
        if self.target_sigma_x_threshold not in [np.inf, None]:
            sigma_x_color = (
                green if abs(cb[1] - tb[1]) < self.target_sigma_x_threshold else red
            )
        img = cv2.putText(
            img,
            f"sx={cb[1]*1e3:.2f}",
            (215, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_x_color,
        )
        mu_y_color = black
        if self.target_mu_y_threshold not in [np.inf, None]:
            mu_y_color = (
                green if abs(cb[2] - tb[2]) < self.target_mu_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"my={cb[2]*1e3:.2f}",
            (415, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            mu_y_color,
        )
        sigma_y_color = black
        if self.target_sigma_y_threshold not in [np.inf, None]:
            sigma_y_color = (
                green if abs(cb[3] - tb[3]) < self.target_sigma_y_threshold else red
            )
        img = cv2.putText(
            img,
            f"sy={cb[3]*1e3:.2f}",
            (615, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            sigma_y_color,
        )

        if mode == "human":
            cv2.imshow("ARES EA", img)
            cv2.waitKey(200)
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def is_beam_on_screen(self):
        """
        Return `True` when the beam is on the screen and `False` when it isn't.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def setup_accelerator(self):
        """
        Prepare the accelerator for use with the environment. Should mostly be used for
        setting up simulations.

        Override with backend-specific imlementation. Optional.
        """

    def get_magnets(self):
        """
        Return the magnet values as a NumPy array in order as the magnets appear in the
        accelerator.

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def set_magnets(self, magnets):
        """
        Set the magnets to the given values.

        The argument `magnets` will be passed as a NumPy array in the order the magnets
        appear in the accelerator.

        When applicable, this method should block until the magnet values are acutally
        set!

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def reset_accelerator(self):
        """
        Code that should set the accelerator up for a new episode. Run when the `reset`
        is called.

        Mostly meant for simulations to switch to a new incoming beam / misalignments or
        similar things.

        Override with backend-specific imlementation. Optional.
        """

    def update_accelerator(self):
        """
        Update accelerator metrics for later use. Use this to run the simulation or
        cache the beam image.

        Override with backend-specific imlementation. Optional.
        """

    def get_beam_parameters(self):
        """
        Get the beam parameters measured on the diagnostic screen as NumPy array grouped
        by dimension (e.g. mu_x, sigma_x, mu_y, sigma_y).

        Override with backend-specific imlementation. Must be implemented!
        """
        raise NotImplementedError

    def get_incoming_parameters(self):
        """
        Get all physical beam parameters of the incoming beam as NumPy array in order
        energy, mu_x, mu_xp, mu_y, mu_yp, sigma_x, sigma_xp, sigma_y, sigma_yp, sigma_s,
        sigma_p.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_misalignments(self):
        """
        Get misalignments of the quadrupoles and the diagnostic screen as NumPy array in
        order AREAMQZM1.misalignment.x, AREAMQZM1.misalignment.y,
        AREAMQZM2.misalignment.x, AREAMQZM2.misalignment.y, AREAMQZM3.misalignment.x,
        AREAMQZM3.misalignment.y, AREABSCR1.misalignment.x, AREABSCR1.misalignment.y.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_screen_image(self):
        """
        Retreive the beam image as a 2-dimensional NumPy array.

        Note that if reading the beam image is expensive, it is best to cache the image
        in the `update_accelerator` method and the read the cached variable here.

        Ideally, the pixel values should look somewhat similar to the 12-bit values from
        the real screen camera.

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_binning(self):
        """
        Return binning currently set on the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_screen_resolution(self):
        """
        Return (binned) resolution of the screen camera as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_pixel_size(self):
        """
        Return the (binned) size of the area on the diagnostic screen covered by one
        pixel as NumPy array [x, y].

        Override with backend-specific imlementation. Optional.
        """
        raise NotImplementedError

    def get_accelerator_observation_space(self):
        """
        Return a dictionary of aditional observation spaces for observations from the
        accelerator backend, e.g. incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_observation(self):
        """
        Return a dictionary of aditional observations from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}

    def get_accelerator_info(self):
        """
        Return a dictionary of aditional info from the accelerator backend, e.g.
        incoming beam and misalignments in simulation.

        Override with backend-specific imlementation. Optional.
        """
        return {}


class ARESEACheetah(ARESEA):
    def __init__(
        self,
        incoming_mode="constant",
        incoming_values=np.array([80e6,-5e-4,6e-5,3e-4,3e-5,4e-4,4e-5,1e-4,4e-5,0,4e-6]),
        misalignment_mode="constant",
        misalignment_values=np.zeros(8),
        abort_if_off_screen=False,
        action_mode="delta",
        include_screen_image_in_info=False,
        magnet_init_mode="constant",
        magnet_init_values=[10,-10,0,10,0],
        reward_mode="negative_objective",
        target_beam_mode="random",
        target_beam_values=None,
        target_mu_x_threshold=3.3198e-6,
        target_mu_y_threshold=2.4469e-6,
        target_sigma_x_threshold=3.3198e-6,
        target_sigma_y_threshold=2.4469e-6,
        threshold_hold=1,
        time_reward=-0.0,
    ):
        super().__init__(
            action_mode=action_mode,
            abort_if_off_screen=abort_if_off_screen,
            include_screen_image_in_info=include_screen_image_in_info,
            magnet_init_mode=magnet_init_mode,
            magnet_init_values=magnet_init_values,
            reward_mode=reward_mode,
            target_beam_mode=target_beam_mode,
            target_beam_values=target_beam_values,
            target_mu_x_threshold=target_mu_x_threshold,
            target_mu_y_threshold=target_mu_y_threshold,
            target_sigma_x_threshold=target_sigma_x_threshold,
            target_sigma_y_threshold=target_sigma_y_threshold,
            threshold_hold=threshold_hold,
            time_reward=time_reward,
        )

        self.incoming_mode = incoming_mode
        self.incoming_values = incoming_values
        self.misalignment_mode = misalignment_mode
        self.misalignment_values = misalignment_values

        # Create particle simulation
        with open("utils/lattice.pkl", "rb") as f:
            self.simulation = pickle.load(f)

    def is_beam_on_screen(self):
        screen = self.simulation.AREABSCR1
        beam_position = np.array([screen.read_beam.mu_x, screen.read_beam.mu_y])
        limits = np.array(screen.resolution) / 2 * np.array(screen.pixel_size)
        extended_limits = (
            limits + np.array([screen.read_beam.sigma_x, screen.read_beam.sigma_y]) * 2
        )
        return np.all(np.abs(beam_position) < extended_limits)

    def get_magnets(self):
        return np.array(
            [
                self.simulation.AREAMQZM1.k1,
                self.simulation.AREAMQZM2.k1,
                self.simulation.AREAMCVM1.angle,
                self.simulation.AREAMQZM3.k1,
                self.simulation.AREAMCHM1.angle,
            ]
        )

    def set_magnets(self, magnets):
        self.simulation.AREAMQZM1.k1 = magnets[0]
        self.simulation.AREAMQZM2.k1 = magnets[1]
        self.simulation.AREAMCVM1.angle = magnets[2]
        self.simulation.AREAMQZM3.k1 = magnets[3]
        self.simulation.AREAMCHM1.angle = magnets[4]

    def reset_accelerator(self):
        # New domain randomisation
        if self.incoming_mode == "constant":
            incoming_parameters = self.incoming_values
        elif self.incoming_mode == "random":
            incoming_parameters = self.observation_space["incoming"].sample()
        else:
            raise ValueError(f'Invalid value "{self.incoming_mode}" for incoming_mode')
        self.incoming = cheetah.ParameterBeam.from_parameters(
            energy=incoming_parameters[0],
            mu_x=incoming_parameters[1],
            mu_xp=incoming_parameters[2],
            mu_y=incoming_parameters[3],
            mu_yp=incoming_parameters[4],
            sigma_x=incoming_parameters[5],
            sigma_xp=incoming_parameters[6],
            sigma_y=incoming_parameters[7],
            sigma_yp=incoming_parameters[8],
            sigma_s=incoming_parameters[9],
            sigma_p=incoming_parameters[10],
        )

        if self.misalignment_mode == "constant":
            misalignments = self.misalignment_values
        elif self.misalignment_mode == "random":
            misalignments = self.observation_space["misalignments"].sample()
        else:
            raise ValueError(
                f'Invalid value "{self.misalignment_mode}" for misalignment_mode'
            )
        self.simulation.AREAMQZM1.misalignment = misalignments[0:2]
        self.simulation.AREAMQZM2.misalignment = misalignments[2:4]
        self.simulation.AREAMQZM3.misalignment = misalignments[4:6]
        self.simulation.AREABSCR1.misalignment = misalignments[6:8]

    def update_accelerator(self):
        self.simulation(self.incoming)

    def get_beam_parameters(self):
        return np.array(
            [
                self.simulation.AREABSCR1.read_beam.mu_x,
                self.simulation.AREABSCR1.read_beam.sigma_x,
                self.simulation.AREABSCR1.read_beam.mu_y,
                self.simulation.AREABSCR1.read_beam.sigma_y,
            ]
        )

    def get_incoming_parameters(self):
        # Parameters of incoming are typed out to guarantee their order, as the
        # order would not be guaranteed creating np.array from dict.
        return np.array(
            [
                self.incoming.energy,
                self.incoming.mu_x,
                self.incoming.mu_xp,
                self.incoming.mu_y,
                self.incoming.mu_yp,
                self.incoming.sigma_x,
                self.incoming.sigma_xp,
                self.incoming.sigma_y,
                self.incoming.sigma_yp,
                self.incoming.sigma_s,
                self.incoming.sigma_p,
            ]
        )

    def get_misalignments(self):
        return np.array(
            [
                self.simulation.AREAMQZM1.misalignment[0],
                self.simulation.AREAMQZM1.misalignment[1],
                self.simulation.AREAMQZM2.misalignment[0],
                self.simulation.AREAMQZM2.misalignment[1],
                self.simulation.AREAMQZM3.misalignment[0],
                self.simulation.AREAMQZM3.misalignment[1],
                self.simulation.AREABSCR1.misalignment[0],
                self.simulation.AREABSCR1.misalignment[1],
            ],
            dtype=np.float32,
        )

    def get_screen_image(self):
        # Beam image to look like real image by dividing by goodlooking number and
        # scaling to 12 bits
        return self.simulation.AREABSCR1.reading / 1e9 * 2**12

    def get_binning(self):
        return np.array(self.simulation.AREABSCR1.binning)

    def get_screen_resolution(self):
        return np.array(self.simulation.AREABSCR1.resolution) / self.get_binning()

    def get_pixel_size(self):
        return np.array(self.simulation.AREABSCR1.pixel_size) * self.get_binning()

    def get_accelerator_observation_space(self):
        return {
            "incoming": spaces.Box(
                low=np.array(
                    [
                        80e6,
                        -1e-3,
                        -1e-4,
                        -1e-3,
                        -1e-4,
                        1e-5,
                        1e-6,
                        1e-5,
                        1e-6,
                        1e-6,
                        1e-4,
                    ],
                    dtype=np.float32,
                ),
                high=np.array(
                    [160e6, 1e-3, 1e-4, 1e-3, 1e-4, 5e-4, 5e-5, 5e-4, 5e-5, 5e-5, 1e-3],
                    dtype=np.float32,
                ),
            ),
            "misalignments": spaces.Box(low=-2e-3, high=2e-3, shape=(8,)),
        }

    def get_accelerator_observation(self):
        return {
            "incoming": self.get_incoming_parameters(),
            "misalignments": self.get_misalignments(),
        }


def read_from_yaml(path):
    with open(f"{path}.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def save_to_yaml(data, path):
    with open(f"{path}.yaml", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    main()
