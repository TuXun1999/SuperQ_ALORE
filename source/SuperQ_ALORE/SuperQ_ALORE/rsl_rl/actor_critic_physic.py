# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import csv
import os
from datetime import datetime

# TODO: libraries used by ALORE (for visualization purpose perhaps?)
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA


import torch
import torch.nn as nn
from torch.distributions import Normal
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic
from .physic_estimator import PhysicEstimator
from rsl_rl.utils import resolve_nn_activation
from .interactive_gnn import InteractiveGNN 


class PhysicActorCritic(ActorCritic):

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions,
        object_types = None,  # New argument to specify object types for GNN processing
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        physic_estimation_enabled=True,
        GNN_enabled=False,
        GNN_obj_enabled=False, # Extend the GNN with object-shape-related nodes
        **kwargs,
    ):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs
        )

        # Actor policy currently uses one-step observations.
        self.history_length = 1
        
        if not hasattr(self, 'device'):
            self.device = kwargs.get('device', 'cuda:0')

        
        self.obs_groups = obs_groups
        self.object_types = object_types
        
        # Specify the dims for the MLPs
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]
        
        # Actor obs (per environment)
        self.num_actor_obs = int(num_actor_obs / self.history_length)  
        
        activation = resolve_nn_activation(activation)

        ## The ablation study on velocity estimation & GNN features
        self.physic_estimation_enabled = physic_estimation_enabled
        self.GNN_enabled = GNN_enabled
        self.GNN_obj_enabled = GNN_obj_enabled
        
        # The input to the actor predictor consists of raw env observations and the 
        # the graph neural network output
        
        if GNN_enabled:
            mlp_input_dim_a = num_actor_obs + 128
        else:
            mlp_input_dim_a = num_actor_obs 

        mlp_input_dim_c = num_critic_obs

        # Multi-head actor policy
        shared_layers = []
        shared_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        shared_layers.append(activation)
        for i in range(len(actor_hidden_dims) - 1):
            shared_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i+1]))
            shared_layers.append(activation)
        self.shared_mlp = nn.Sequential(*shared_layers)

        # Base control head
        self.base_head = nn.Linear(actor_hidden_dims[-1], 3)
        # Arm control head
        self.arm_head = nn.Linear(actor_hidden_dims[-1], 6)

        # Configure estimator input from dedicated com_estimation group when available.
        estimator_input_dim = self.num_actor_obs
        estimator_history_length = 10
        if "com_estimation" in obs.keys():
            est_obs = obs["com_estimation"]
            if len(est_obs.shape) == 3:
                estimator_history_length = int(est_obs.shape[1])
                estimator_input_dim = int(est_obs.shape[2])
            elif len(est_obs.shape) == 2:
                estimator_input_dim = int(est_obs.shape[1])

        # Add a physic estimator
        if physic_estimation_enabled:
            self.physic_estimator = PhysicEstimator(
                input_dim = estimator_input_dim,
                output_dim=3,  # [x, y, z] # COM
                device=self.device,
                history_length=estimator_history_length
            )
            print(f'Estimator: {self.physic_estimator}')
        else:
            self.physic_estimator = None

        ## add the interactive GNN
        if GNN_enabled:
            print("Interactive GNN enabled in ActorCritic")
            self.interactive_gnn = InteractiveGNN(
                node_dim=15, # node feature dimension
                edge_dim=7,  # edge feature dimension
                hidden_dim=64,
                out_dim=128
            )
            if GNN_obj_enabled:
                print("Object-shape-related nodes enabled in Interactive GNN")
                self.interactive_gnn.build_obj_node_info(self.object_types)
        else:            
            print("Interactive GNN disabled in ActorCritic")
            self.interactive_gnn = None

        self.num_one_step_obs = num_actor_obs  # Number of observations used for one-step prediction


    """
    The following contents are only about PPO
    """
    def update_distribution(self, observations, critic_observations, object_type = None):
        B = observations.shape[0]
        T = self.history_length
        D = self.num_actor_obs
        obs_seq = observations.view(B, T, D)  # (B, T, D)

        obs_augmented = obs_seq  # (B, T, D) -- ablation without velocity prediction, to test the effect of GNN features alone
        
        ## Ablation study: interactive GNN processing
        if self.GNN_enabled:
            if self.GNN_obj_enabled:
                node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations, object_type)
            else:
                node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations)
            z = self.interactive_gnn(node_features, edge_index, edge_attr, batch)  # shape: [B, 128]
            actor_input = torch.cat([obs_augmented.reshape(B, -1), z], dim=-1)
        else:
            actor_input = obs_augmented.reshape(B, -1)  # If GNN is disabled, just use the augmented observations
        
        # Feed the input to the MLPs
        shared_feat = self.shared_mlp(actor_input)
        base_mean = self.base_head(shared_feat)
        arm_mean = self.arm_head(shared_feat)
        mean = torch.cat([base_mean, arm_mean], dim=-1)
        
        # Pad the actions to match the 12D action space accepted by the environment
        mean = torch.cat([mean, torch.zeros(mean.shape[0], 3, device=mean.device)], dim=-1)
        
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)


    def act(self, obs, **kwargs):
        # Separate obs & critic_obs
        observations = obs["policy"]
        critic_observations = obs["critic"]
        object_type = obs["object_idx"][:, 0].long().squeeze()  # Assuming object type is represented as an integer index in the observation
        """
        actions: base velocity (3) + arm joint (7) + base pose (2: pitch, height)
        (Forced to match 12D action space of the pretrained locomotion policy)
        
        The last dimensions are forced to be 0
        """
        self.update_distribution(observations, critic_observations, object_type)
        try:
            actions_raw = self.distribution.sample() 
        except Exception as e:
            print(f"Error during action sampling: {e}")
            print(f"Mean: {self.distribution.mean[0]}")
            print(f"Std: {self.distribution.stddev[0]}")
            raise e

        return actions_raw


    def act_inference(self, obs, **kwargs):
        # Separate obs out
        observations = obs["policy"]
        critic_observations = obs["critic"]
        object_type = obs["object_idx"][:, 0].long().squeeze()  # Assuming object type is represented as an integer index in the observation
        
        # Reshape the observations to (B, T, D)
        B = observations.shape[0]
        T = self.history_length
        D = self.num_actor_obs
        obs_seq = observations.view(B, T, D)  # (B, T, D)

        # The estimation of object physics is done separately
        obs_augmented = obs_seq  # (B, T, D) -- ablation without velocity prediction, to test the effect of GNN features alone
        
        # interactive GNN processing
        if self.GNN_enabled:
            # Two cases: object-shape-related nodes enabled or not
            if self.GNN_obj_enabled:
                node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations, object_type)
            else:
                node_features, edge_index, edge_attr, batch = self.interactive_gnn.build_interaction_graph(obs_seq, critic_observations)
            z = self.interactive_gnn(node_features, edge_index, edge_attr, batch)  # shape: [B, 128]
            # Concatenate the augmented observations and GNN output for action prediction
            actor_input = torch.cat([obs_augmented.reshape(B, -1), z], dim=-1)
        else:
            actor_input = obs_augmented.reshape(B, -1)  # If GNN is disabled, just use the augmented observations
        
            
        shared_feat = self.shared_mlp(actor_input)
        base_mean = self.base_head(shared_feat)
        arm_mean = self.arm_head(shared_feat)
        actions_mean = torch.cat([base_mean, arm_mean], dim=-1)
        
        # Pad three zeros for the last three dimensions of the action
        actions_mean = torch.cat([actions_mean, torch.zeros(actions_mean.shape[0], 3, device=actions_mean.device)], dim=-1)

        
        return actions_mean
    

    """(DEPRECATED) Saving to csv may not be demanded yet..."""
    def _save_predictions_and_gt_to_csv(self, pred_vx, pred_vy, pred_omega, 
                                    gt_vx, gt_vy, gt_omega):

        csv_file = "plan_vel_predictions_vs_gt_table_square.csv"
        
        # 
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # 
            if not file_exists:
                writer.writerow([
                    'timestamp', 'env_id', 
                    'pred_vx', 'pred_vy', 'pred_omega',
                    'gt_vx', 'gt_vy', 'gt_omega',
                    'error_vx', 'error_vy', 'error_omega'
                ])
            
         
            if hasattr(pred_vx, 'cpu'):
                pred_vx = pred_vx.cpu().numpy()
                pred_vy = pred_vy.cpu().numpy()
                pred_omega = pred_omega.cpu().numpy()
                gt_vx = gt_vx.cpu().numpy()
                gt_vy = gt_vy.cpu().numpy()
                gt_omega = gt_omega.cpu().numpy()
            
        
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            for env_id in range(len(pred_vx)):
   
                error_vx = pred_vx[env_id] - gt_vx[env_id]
                error_vy = pred_vy[env_id] - gt_vy[env_id]
                error_omega = pred_omega[env_id] - gt_omega[env_id]
                
                writer.writerow([
                    current_time,
                    env_id,
                    pred_vx[env_id],
                    pred_vy[env_id], 
                    pred_omega[env_id],
                    gt_vx[env_id],
                    gt_vy[env_id],
                    gt_omega[env_id],
                    error_vx,
                    error_vy,
                    error_omega
                ])


    """(DEPRECATED) Visualization may not be demanded yet..."""
    def visualize_gnn_features_pca_only(
        self,
        z,
        labels=None,
        save_path='gnn_pca_visualization.png',
        scale_mode='none',          
        crop_mode='none',           # 
        crop_lo=2, crop_hi=98,      
        margin_ratio=0.06,          
        symlog_linthresh=1e-3,      
        break_x=((0.0, 0.3),),      
        break_y=((-0.05, 0.0),)     
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        from scipy.stats import chi2
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # --- ---
        z_np = z.cpu().detach().numpy() if hasattr(z, 'cpu') else np.array(z)
        if z_np.shape[0] < 5:
            print(f"Too few samples ({z_np.shape[0]}) for visualization")
            return

        # --- ---
        def draw_confidence_ellipse(ax, x, y, color, alpha=0.3, confidence=0.95):
            if len(x) < 3: 
                return
            try:
                mean = np.array([np.mean(x), np.mean(y)])
                cov  = np.cov(x, y)
                eigvals, eigvecs = np.linalg.eigh(cov + 1e-12*np.eye(2))
                order = np.argsort(eigvals)[::-1]
                eigvals, eigvecs = eigvals[order], eigvecs[:, order]
                chi2_val = chi2.ppf(confidence, df=2)
                width  = 2*np.sqrt(chi2_val*eigvals[0])
                height = 2*np.sqrt(chi2_val*eigvals[1])
                angle  = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
                ell = Ellipse(mean, width, height, angle=angle,
                            facecolor=color, edgecolor=color, linewidth=2, alpha=alpha)
                ax.add_patch(ell)
            except Exception as e:
                print(f"Failed to draw ellipse: {e}")

        # ------
        pca = PCA(n_components=2, random_state=42)
        z_pca = pca.fit_transform(z_np)
        explained_var = pca.explained_variance_ratio_

        z_plot = z_pca.copy()
        if scale_mode == 'zscore':
            z_plot = StandardScaler().fit_transform(z_plot)

        print(f"PCA data range: X[{z_plot[:, 0].min():.3f}, {z_plot[:, 0].max():.3f}], Y[{z_plot[:, 1].min():.3f}, {z_plot[:, 1].max():.3f}]")

        object_names = ['Table1', 'Table2', 'Chair']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # ----
        if crop_mode == 'break':
            try:
                from brokenaxes import brokenaxes
                
     
                x_min, x_max = z_plot[:, 0].min(), z_plot[:, 0].max()
                y_min, y_max = z_plot[:, 1].min(), z_plot[:, 1].max()
                
                print(f"Setting up broken axes with data range: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}]")
                

                xlims = [(x_min - 0.1, break_x[0][0]), (break_x[0][1], x_max + 0.1)]
                ylims = [(y_min - 0.1, break_y[0][0]), (break_y[0][1], y_max + 0.1)]
                
                fig = plt.figure(figsize=(12, 10))
                bax = brokenaxes(
                    xlims=xlims,
                    ylims=ylims,
                    hspace=0.05, 
                    wspace=0.05,
                    despine=False
                )
                scatter_ax = bax
                
                print("Broken axes created successfully")
                
            except ImportError:
                print("brokenaxes not available, falling back to regular plot")
                crop_mode = 'none'
                plt.figure(figsize=(10, 8))
                scatter_ax = plt.gca()
            except Exception as e:
                print(f"Failed to create broken axes: {e}, falling back to regular plot")
                crop_mode = 'none'
                plt.figure(figsize=(10, 8))
                scatter_ax = plt.gca()
        else:
            plt.figure(figsize=(10, 8))
            scatter_ax = plt.gca()


        if labels is not None:
            unique_labels = np.unique(labels)
            for i, lab in enumerate(unique_labels):
                m = (labels == lab)
                color = colors[i % len(colors)]

                scatter_ax.scatter(z_plot[m, 0], z_plot[m, 1],
                        c=color, label=object_names[i % len(object_names)],
                        alpha=0.85, s=80, edgecolors='black', linewidth=0.6)
                
                if np.sum(m) >= 3:
                    draw_confidence_ellipse(scatter_ax, z_plot[m, 0], z_plot[m, 1], color)
                    
                    cx, cy = z_plot[m, 0].mean(), z_plot[m, 1].mean()
                    scatter_ax.scatter(cx, cy, c='white', s=120, edgecolors=color,
                            linewidth=2, marker='x', zorder=10)
        else:
            scatter_ax.scatter(z_plot[:, 0], z_plot[:, 1],
                            alpha=0.8, s=70, edgecolors='black', linewidth=0.4)

        if crop_mode == 'percentile':
            xlo, xhi = np.percentile(z_plot[:, 0], [crop_lo, crop_hi])
            ylo, yhi = np.percentile(z_plot[:, 1], [crop_lo, crop_hi])
            xr, yr = xhi - xlo, yhi - ylo
            scatter_ax.set_xlim(xlo - xr*margin_ratio, xhi + xr*margin_ratio)
            scatter_ax.set_ylim(ylo - yr*margin_ratio, yhi + yr*margin_ratio)
        elif crop_mode == 'tight':
            xlo, xhi = z_plot[:, 0].min(), z_plot[:, 0].max()
            ylo, yhi = z_plot[:, 1].min(), z_plot[:, 1].max()
            xr, yr = xhi - xlo, yhi - ylo
            scatter_ax.set_xlim(xlo - xr*margin_ratio, xhi + xr*margin_ratio)
            scatter_ax.set_ylim(ylo - yr*margin_ratio, yhi + yr*margin_ratio)

        if scale_mode == 'symlog':
            try:
                scatter_ax.set_xscale('symlog', linthresh=symlog_linthresh)
                scatter_ax.set_yscale('symlog', linthresh=symlog_linthresh)
            except:
                print("symlog scale not supported with broken axes")

        if crop_mode != 'break':   
            scatter_ax.set_title(
                'GNN Feature PCA Visualization with 95% Confidence Ellipses\n'
                f'Explained Variance: {explained_var.sum():.3f}',
                fontsize=14, fontweight='bold', pad=18
            )
            scatter_ax.set_xlabel(f'PC1 ({explained_var[0]:.3f})', fontsize=12)
            scatter_ax.set_ylabel(f'PC2 ({explained_var[1]:.3f})', fontsize=12)
            scatter_ax.grid(True, alpha=0.3)
        else:
            fig.suptitle(
                'GNN Feature PCA Visualization with 95% Confidence Ellipses\n'
                f'Explained Variance: {explained_var.sum():.3f}',
                fontsize=14, fontweight='bold', y=0.95
            )

        if labels is not None:
            if crop_mode != 'break':
                scatter_ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=False)
            else:
                fig.legend(labels=[object_names[i] for i in range(len(np.unique(labels)))], 
                          loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"PCA visualization saved to {save_path}")

        return {'pca_components': z_pca, 'explained_variance': explained_var}
