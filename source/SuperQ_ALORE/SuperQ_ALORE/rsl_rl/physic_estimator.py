import torch
import torch.nn as nn
import torch.optim as optim


class PhysicEstimator(nn.Module):
    def __init__(self,
                 input_dim=44,
                 output_dim=3,  # [object_x, object_y, object_ang_vel_z]
                 lstm_hidden_size=128,
                 lstm_layers=1,
                 mlp_hidden_dim=64,
                 learning_rate=1e-3,
                 max_grad_norm=10.0,
                 history_length=10, # Assuming the history length is 10
                 device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.num_actor_obs = input_dim  # Number of observations used for one-step prediction
        self.history_length = history_length 

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size = input_dim,
            hidden_size = lstm_hidden_size,
            num_layers = lstm_layers,
            batch_first = True
        )

        # Output heads (HistoryEncoder-style): predict mean and log-variance.
        self.output_backbone = nn.Sequential(
            nn.Linear(lstm_hidden_size, mlp_hidden_dim),
            nn.ReLU(),
        )
        self.to_mean = nn.Linear(mlp_hidden_dim, output_dim)
        self.to_log_var = nn.Linear(mlp_hidden_dim, output_dim)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.GaussianNLLLoss(full=False, reduction="mean", eps=1e-6)
        self.max_grad_norm = max_grad_norm

        self.to(self.device)

        self.num_one_step_obs = input_dim  # Number of observations used for one-step prediction
        
        print(f"PhysicEstimator initialized with input_dim={input_dim}, output_dim={output_dim}, ")


    def _prepare_sequence(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Normalize estimator input to [B, T, D] while preserving LSTM backbone usage."""
        if obs_history.dim() == 3:
            return obs_history

        if obs_history.dim() == 2:
            batch_size, feat_dim = obs_history.shape
            flat_dim = self.history_length * self.num_actor_obs
            if feat_dim == flat_dim:
                return obs_history.view(batch_size, self.history_length, self.num_actor_obs)
            if feat_dim == self.num_actor_obs:
                return obs_history.unsqueeze(1)

        raise ValueError(
            "Unexpected obs_history shape for PhysicEstimator: "
            f"{tuple(obs_history.shape)}. Expected [B, T, D], [B, D], or [B, T*D]."
        )

    def forward(self, obs_history):
        """
        obs_history: (B, T, D)
        Returns: (B, 3) as the predicted [object_x, object_y, object_ang_vel_z]
        """
        
        obs_history = self._prepare_sequence(obs_history)
        # print(f"====obs_history shape: {obs_history.shape}====")
        lstm_out, (h_n, _) = self.lstm(obs_history)  # h_n: (num_layers, B, H)
        # print(f"====h_n shape: {h_n.shape}====")
        last_hidden = h_n[-1]  # (B, H)
        # print(f"====last_hidden shape: {last_hidden.shape}====")
        feat = self.output_backbone(last_hidden)
        mean = self.to_mean(feat)
        log_var = torch.clamp(self.to_log_var(feat), min=-10, max=5)
        return mean, log_var
    

    def update(self, obs_history, critic_obs=None, target_com=None):
        """
        obs_history: (B, T, D) or (B, D)
        target_com: optional (B, 3), explicit supervised target for CoM.
        If target_com is None, fallback to legacy target extraction from critic_obs.
        """
        if target_com is None:
            if critic_obs is None:
                raise ValueError("Either target_com or critic_obs must be provided to PhysicEstimator.update().")
            obj_ang_vel_z_gt = critic_obs[:, -4].detach()
            obj_lin_vel_x_gt = critic_obs[:, -9].detach()
            obj_lin_vel_y_gt = critic_obs[:, -8].detach()
            y = torch.stack([obj_lin_vel_x_gt, obj_lin_vel_y_gt, obj_ang_vel_z_gt], dim=-1)
        else:
            if target_com.shape[-1] != 3:
                raise ValueError(f"Expected target_com last dim == 3, got {target_com.shape}.")
            y = target_com.detach()

        x = obs_history.to(self.device)
        y = y.to(self.device)

        self.train()
        mean, log_var = self.forward(x)
        var = torch.exp(log_var).clamp(min=1e-5, max=150)
        loss = self.loss_fn(mean, y, var)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()


    def predict(self, obs_history):
        """
        obs_history: (B, T, D)
        Returns: (B, 3) as numpy array
        """
        if not isinstance(obs_history, torch.Tensor):
            obs_history = torch.tensor(obs_history, dtype=torch.float32)

        x = obs_history.to(self.device)

        self.eval()
        with torch.no_grad():
            y_pred, _ = self.forward(x)

        return y_pred.cpu().numpy()
