from typing import Iterator, Tuple, Any

import glob
import joblib
import numpy as np
import os

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class RptDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for RPT dataset."""

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
                        'left_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Left camera RGB observation.',
                        ),
                        'right_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Right camera RGB observation.',
                        ),
                        'joint_pos': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='xArm joint positions (7 DoF).',
                        ),
                        'gripper': tfds.features.Scalar(
                            dtype=np.bool_,
                            doc='Binary gripper state (1 - closed, 0 - open)',
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

            ep_obs = []
            for j, obs_file in enumerate(sorted(os.listdir(episode_path))):
                obs_path = os.path.join(episode_path, obs_file)
                if not obs_file.endswith(".pkl"):
                    continue
                with open(obs_path, "rb") as f:
                    obs = joblib.load(f)

                ep_obs.append({
                    "hand_image": obs["rgb_hand"],
                    "left_image": obs["rgb_left"],
                    "right_image": obs["rgb_right"],
                    "joint_pos": np.array(obs["joint_pos"]).astype(np.float32),
                    "gripper": 1 - obs["gripper_width" if "gripper_width" in obs else "gripper_closed"]
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
                        'left_image': s_t['left_image'],
                        'right_image': s_t['right_image'],
                        'joint_pos': s_t['joint_pos'],
                        'gripper': s_t['gripper'],
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

        root_path = "/data1/ilija/rpt"

        dirs = [
            ("destack-red-gray-cube_05-22-2023", "destack cube"),
            ("pick-bin_05-23-2023", "pick an object from the bin"),
            ("pick-yellow-cube_01-23-2023", "pick yellow cube"),
            ("stack-green-blue-cube_05-20-2023", "stack cube")
        ]

        dir_paths = [(os.path.join(root_path, d), li) for (d, li) in dirs]
        episode_paths = [(ep_path, li) for (dir_path, li) in dir_paths for ep_path in glob.glob(f"{dir_path}/*")]

        # for smallish datasets, use single-thread parsing
        for ep_path, li in episode_paths:
            yield _parse_example(ep_path, li)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

