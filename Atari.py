import atari_py
import os
from gym.utils import seeding
import numpy as np


def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class Atari:

    def __init__(self, game='pong', obs_type='ram_image', frameskip=[2,5], repeat_action_probability=0.):


        assert obs_type in ('ram','image','ram_image')

        self.ale = atari_py.ALEInterface()

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist' % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip

        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()

        self._legal_action_set = self.ale.getMinimalActionSet()
        self.viewer=None

    @property
    def action_len(self):
        return len(self._legal_action_set)

    @property
    def legal_actions(self):
        return self._legal_action_set


    def step(self, a):
            reward = 0.0
            action = self._legal_action_set[a]

            if isinstance(self.frameskip, int):
                num_steps = self.frameskip
            else:
                num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
            for _ in range(num_steps):
                reward += self.ale.act(action)
            ob = self._get_obs()

            return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    def _get_obs(self):
      # return a dictionary "[ram,img]"
        return {"ram":self._get_ram(),"image":self._get_image()}

    def seed(self, seed=None):
            self.np_random, seed1 = seeding.np_random(seed)
            # Derive a random seed. This gets passed as a uint, but gets
            # checked as an int elsewhere, so we need to keep it below
            # 2**31.
            seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
            # Empirically, we need to seed before loading the ROM.
            self.ale.setInt(b'random_seed', seed2)
            self.ale.loadROM(self.game_path)

            return [seed1, seed2]

    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
            return [ACTION_MEANING[i] for i in self._legal_action_set]

    def get_keys_to_action(self):
            KEYWORD_TO_KEY = {
                'UP': ord('w'),
                'DOWN': ord('s'),
                'LEFT': ord('a'),
                'RIGHT': ord('d'),
                'FIRE': ord(' '),
            }

            keys_to_action = {}

            for action_id, action_meaning in enumerate(self.get_action_meanings()):
                keys = []
                for keyword, key in KEYWORD_TO_KEY.items():
                    if keyword in action_meaning:
                        keys.append(key)
                keys = tuple(sorted(keys))

                assert keys not in keys_to_action
                keys_to_action[keys] = action_id

            return keys_to_action

    def clone_state(self):
            """Clone emulator state w/o system state. Restoring this state will
            *not* give an identical environment. For complete cloning and restoring
            of the full state, see `{clone,restore}_full_state()`."""
            state_ref = self.ale.cloneState()
            state = self.ale.encodeState(state_ref)
            self.ale.deleteState(state_ref)
            return state

    def restore_state(self, state):
            """Restore emulator state w/o system state."""
            state_ref = self.ale.decodeState(state)
            self.ale.restoreState(state_ref)
            self.ale.deleteState(state_ref)

    def clone_full_state(self):
            """Clone emulator state w/ system state including pseudorandomness.
            Restoring this state will give an identical environment."""
            state_ref = self.ale.cloneSystemState()
            state = self.ale.encodeState(state_ref)
            self.ale.deleteState(state_ref)
            return state

    def restore_full_state(self, state):
            """Restore emulator state w/ system state including pseudorandomness."""
            state_ref = self.ale.decodeState(state)
            self.ale.restoreSystemState(state_ref)
            self.ale.deleteState(state_ref)

ACTION_MEANING = {
        0: "NOOP",
        1: "FIRE",
        2: "UP",
        3: "RIGHT",
        4: "LEFT",
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT",
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE",
    }