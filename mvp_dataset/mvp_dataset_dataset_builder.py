from typing import Iterator, Tuple, Any

import glob
import numpy as np
import os
import pickle

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from scipy.spatial.transform import Rotation as R


class MvpDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for MVP dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'hand_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Hand camera RGB observation.',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='xArm joint positions (7 DoF).',
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Binary gripper state (1 - closed, 0 - open)',
                        ),
                        'pose': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Gripper pose, robot frame, [3 position, 4 rotation]'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7 delta joint pos,'
                            '1x gripper binary state].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path, language_instruction):

            # load raw data
            ep_obs = []
            for j, obs_file in enumerate(sorted(os.listdir(episode_path))):
                obs_path = os.path.join(episode_path, obs_file)
                with open(obs_path, "rb") as f:
                    obs = pickle.load(f)
                ep_obs.append({
                    "hand_image": obs["rgb" if "rgb" in obs else "rgb_ego"],
                    "joint_pos": obs["joint_positions"][:7].astype(np.float32),
                    "gripper": 1 - obs["gripper_open"],
                    "pose": np.concatenate([
                        obs["T_ee_in_link0"][:3, 3],
                        R.from_matrix(obs['T_ee_in_link0'][:3, :3]).as_quat()
                    ]).astype(np.float32)
                })

            # compute Kona language embedding
            language_embedding = self._embed([language_instruction])[0].numpy()

            # assemble episode
            episode = []
            for t, s_t in enumerate(ep_obs):
                # compute the action
                s_t1 = ep_obs[t + 1] if t < len(ep_obs) - 1 else s_t
                action = np.concatenate([
                    s_t1["joint_pos"] - s_t["joint_pos"],
                    np.array([(s_t1["gripper"])], dtype=np.float32)
                ])

                episode.append({
                    'observation': {
                        'hand_image': s_t['hand_image'],
                        'joint_pos': s_t['joint_pos'],
                        'gripper': s_t['gripper'],
                        'pose': s_t['pose']
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': float(t == (len(ep_obs) - 1)),
                    'is_first': t == 0,
                    'is_last': t == (len(ep_obs) - 1),
                    'is_terminal': t == (len(ep_obs) - 1),
                    'language_instruction': language_instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        root_path = "/data1/ilija/mvp-real/real_demos/"

        tasks = [
            "close-single-fridge-door_2022-06-09+10",
            "pick-cube_2022-06-10",
            "pick-fruit-8x10_2022-06-07",
            "pick-sink_2022-06-09",
            "push-cube_2022-06-11",
            "reach-block_2022-06-11"
        ]

        language_instructions = [
            "close fridge door",
            "pick yellow cube",
            "pick fruit",
            "pick detergent from the sink",
            "push wooden cube",
            "reach red block"
        ]

        task_paths = [os.path.join(root_path, task) for task in tasks]
        episode_paths = [ep_path for task_path in task_paths for ep_path in glob.glob(f"{task_path}/*")]
        assert len(episode_paths) == len(tasks) * 80

        # for smallish datasets, use single-thread parsing
        for i, sample in enumerate(episode_paths):
            language_instruction = language_instructions[i // 80]
            yield _parse_example(sample, language_instruction)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

