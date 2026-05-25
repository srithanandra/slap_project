import torch
import torch.nn as nn
import numpy as np
import time
from jetson_to_teensy import JetsonTeensyBridge


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=(512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ELU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)

class ModelController:

    def __init__(self, model_path):
        try:
            self.device = torch.device('cpu')
            checkpoint = torch.load(model_path, map_location=self.device)
            # print(checkpoint.keys())
            # for k, v in checkpoint["actor_state_dict"].items():
            #     print(f"{k}: {v.shape}")
            if not isinstance(checkpoint, dict) or 'actor_state_dict' not in checkpoint:
                raise ValueError('Expected an Isaac Lab / rsl_rl checkpoint with actor_state_dict.')

            actor_state_dict = checkpoint['actor_state_dict']
            policy_state_dict = {key: value for key, value in actor_state_dict.items() if key.startswith('mlp.')}
            obs_dim = policy_state_dict['mlp.0.weight'].shape[1]
            action_dim = policy_state_dict['mlp.6.weight'].shape[0]

            self.policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim)
            self.policy.load_state_dict(policy_state_dict, strict=True)
            self.policy.eval()

            self.expected_obs_dim = obs_dim
            self.action_dim = action_dim
            self.bridge_joint_count = 12
            if self.action_dim != self.bridge_joint_count:
                raise ValueError(
                    f'Checkpoint action dimension {self.action_dim} does not match the Teensy bridge '
                    f'joint count {self.bridge_joint_count}.'
                )
        except Exception as e:
            raise SystemExit(f'Failed to load AI model: {e}')

        self.bridge = JetsonTeensyBridge()
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.default_positions = np.zeros(self.action_dim, dtype=np.float32)

    def extract_observation_vector(self, feedback):
        try:
            obs = []
            obs.extend([feedback['roll'], feedback['pitch'], feedback['yaw']])
            obs.extend(feedback['gyro'])
            obs.extend(feedback['accel'])
            obs.extend(feedback['motor_positions'])
            obs.extend(self.last_action.tolist())

            if len(obs) != self.expected_obs_dim:
                raise ValueError(
                    f'Built observation vector with {len(obs)} features, but the checkpoint expects '
                    f'{self.expected_obs_dim}. Use the same Isaac Lab observation terms that were used '
                    f'during training.'
                )
            
            obs_tensor = torch.tensor([obs], dtype=torch.float32, device=self.device)
            return obs_tensor
        except Exception as e:
            print(f'Observation extraction error: Missing key {e}')
            return None

    def run_loop(self, frequency=50):
        dt = 1.0 / frequency
        positions = self.default_positions.tolist()
        torques = [0.5] * self.action_dim

        try:
            while True:
                start_time = time.time()
                feedback = self.bridge.communicate(positions, torques, estop=False)

                if feedback:
                    obs_tensor = self.extract_observation_vector(feedback)
                    
                    if obs_tensor is not None:
                        with torch.no_grad():
                            action_tensor = self.policy(obs_tensor)
                        
                        actions = action_tensor.cpu().numpy()[0]
                        self.last_action = actions
                        
                        positions = actions.tolist()
                        torques = [2.0] * self.action_dim

                elapsed = time.time() - start_time
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    
        except KeyboardInterrupt:
            self.bridge.communicate([0.0] * self.bridge_joint_count, [0.0] * self.bridge_joint_count, estop=True)
            self.bridge.close()
        except Exception as e:
            print(f'Runtime control loop error: {e}')
            self.bridge.communicate([0.0] * self.bridge_joint_count, [0.0] * self.bridge_joint_count, estop=True)
            self.bridge.close()