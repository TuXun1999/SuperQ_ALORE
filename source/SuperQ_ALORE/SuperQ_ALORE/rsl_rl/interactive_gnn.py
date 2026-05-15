import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class EdgeConvWithEdgeAttr(MessagePassing):
    def __init__(self, mlp):
        super().__init__(aggr='max')  # EdgeConv 
        self.mlp = mlp

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim], edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate x_i (central), x_j (neighbor), and edge_attr
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(msg_input)
    

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        dims = [in_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def quat_inverse(q):
    q_inv = q.clone()
    q_inv[:, :3] = -q_inv[:, :3]
    return q_inv


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return torch.stack([x, y, z, w], dim=-1)


class InteractiveGNN(nn.Module):
    def __init__(self, node_dim=16, edge_dim=6, hidden_dim=64, out_dim=128):
        super().__init__()
        self.edge_mlp1 = MLP(node_dim * 2 + edge_dim, [64], hidden_dim)
        self.edge_mlp2 = MLP(hidden_dim * 2 + edge_dim, [64], hidden_dim)

        self.conv1 = EdgeConvWithEdgeAttr(self.edge_mlp1)
        self.conv2 = EdgeConvWithEdgeAttr(self.edge_mlp2)

        self.readout = MLP(hidden_dim, [64], out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        # print(f"InteractiveGNN input shapes: x={x.shape}, edge_index={edge_index.shape}, edge_attr={edge_attr.shape}, batch={batch.shape}")
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)

        x_pool = global_mean_pool(x, batch)
        z = self.readout(x_pool)
        return z

    
    def build_interaction_graph(self, obs_seq, critic_obs):
        """
            obs_seq: Tensor of shape [num_env, history_len, num_actor_obs]

            node_features: [N_total, node_dim]
            edge_index: [2, E]
            edge_attr: [E, edge_dim]
            batch: [N_total]
        """
        num_env, history_len, num_actor_obs = obs_seq.shape
        obs_now = obs_seq[:, -1]  # ]
        device = obs_now.device

        ## ============ node features ============================================
        # base: orientation 2, angular velocity 3,                         # 5
        # joints: relative_pose_to_base 7， q, default_q, q-default_q, dq, # 11
        # ee: ee_pose_in_base_frame 7, contact_state,                      # 8
        # object:  obj_pose_in_base_frame 7, goal_vel 3,                   # 10 
        # ========================================================================
        
        # base: orientation 2, angular velocity 3
        base_feat = critic_obs[:, 72:77]    # [num_envs, 5]
        
        # The indices of arm joints in one group of joint states:
        arm_joint_indices = [0, 5, 10, 15, 16, 17]
        q_start_idx = 54
        q_default_start_idx = 36
        q_relative_start_idx = 0
        q_velocity_start_idx = 18
        
        # joints: 
        # link: arm_link_sh0, arm_sh0
        # Find the indices of the joint attributes in critic_obs
        joint1_q_idx = q_start_idx + arm_joint_indices[0]
        joint1_q_default_idx = q_default_start_idx + arm_joint_indices[0]
        joint1_q_relative_idx = q_relative_start_idx + arm_joint_indices[0]
        joint1_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[0]
        
        joint1_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 89:96], 
            # q
            critic_obs[:,joint1_q_idx:joint1_q_idx+1], 
            # default_q
            critic_obs[:,joint1_q_default_idx:joint1_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint1_q_relative_idx:joint1_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint1_q_velocity_idx:joint1_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        
        # link: arm_link_sh1, joint2
        joint2_q_idx = q_start_idx + arm_joint_indices[1]
        joint2_q_default_idx = q_default_start_idx + arm_joint_indices[1]
        joint2_q_relative_idx = q_relative_start_idx + arm_joint_indices[1]
        joint2_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[1]
        
        joint2_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 96:103], 
            # q
            critic_obs[:,joint2_q_idx:joint2_q_idx+1], 
            # default_q
            critic_obs[:,joint2_q_default_idx:joint2_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint2_q_relative_idx:joint2_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint2_q_velocity_idx:joint2_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        
        # link: arm_link_el0, joint3
        joint3_q_idx = q_start_idx + arm_joint_indices[2]
        joint3_q_default_idx = q_default_start_idx + arm_joint_indices[2]
        joint3_q_relative_idx = q_relative_start_idx + arm_joint_indices[2]
        joint3_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[2]
        
        joint3_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 103:110], 
            # q
            critic_obs[:,joint3_q_idx:joint3_q_idx+1], 
            # default_q
            critic_obs[:,joint3_q_default_idx:joint3_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint3_q_relative_idx:joint3_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint3_q_velocity_idx:joint3_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        
        # link: arm_link_el1, joint4
        joint4_q_idx = q_start_idx + arm_joint_indices[3]
        joint4_q_default_idx = q_default_start_idx + arm_joint_indices[3]
        joint4_q_relative_idx = q_relative_start_idx + arm_joint_indices[3]
        joint4_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[3]
        
        joint4_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 110:117], 
            # q
            critic_obs[:,joint4_q_idx:joint4_q_idx+1], 
            # default_q
            critic_obs[:,joint4_q_default_idx:joint4_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint4_q_relative_idx:joint4_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint4_q_velocity_idx:joint4_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        

        # link: arm_link_wr0, joint5
        joint5_q_idx = q_start_idx + arm_joint_indices[4]
        joint5_q_default_idx = q_default_start_idx + arm_joint_indices[4]
        joint5_q_relative_idx = q_relative_start_idx + arm_joint_indices[4]
        joint5_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[4]
        
        joint5_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 117:124], 
            # q
            critic_obs[:,joint5_q_idx:joint5_q_idx+1], 
            # default_q
            critic_obs[:,joint5_q_default_idx:joint5_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint5_q_relative_idx:joint5_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint5_q_velocity_idx:joint5_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        
        # link: arm_link_wr1, joint6
        joint6_q_idx = q_start_idx + arm_joint_indices[5]
        joint6_q_default_idx = q_default_start_idx + arm_joint_indices[5]
        joint6_q_relative_idx = q_relative_start_idx + arm_joint_indices[5]
        joint6_q_velocity_idx = q_velocity_start_idx + arm_joint_indices[5]
        
        joint6_feat = torch.cat([
            # link pose in robot frame
            critic_obs[:, 124:131], 
            # q
            critic_obs[:,joint6_q_idx:joint6_q_idx+1], 
            # default_q
            critic_obs[:,joint6_q_default_idx:joint6_q_default_idx+1], 
            # q_relative = q - default_q
            critic_obs[:, joint6_q_relative_idx:joint6_q_relative_idx+1], 
            # q_velocity
            critic_obs[:, joint6_q_velocity_idx:joint6_q_velocity_idx+1]], 
        dim=-1)  # [num_envs, 11]
        
        # End-effector features: end-effector pose (7) + contact state (1)
        ee_feat = critic_obs[:, 131:139]  # [num_envs, 8]                                  
        
        # Object features:  
        # object pose in base frame (7) + goal_vel (vx, vy, vomega) (3)
        # # TODO: modify it with our own shape encoder
        object_feat = torch.cat([critic_obs[:, 139:146], critic_obs[:, 86:89]], dim=-1)  # [num_envs, 10]  

        
        # Padding to ensure all node features have the same dimension
        # zero-padding → [11]
        num_envs = critic_obs.shape[0]
        base_feat_padded  = torch.cat([base_feat, torch.zeros(num_envs, 6, device=device)], dim=-1) # 
        ee_feat_padded  = torch.cat([ee_feat, torch.zeros(num_envs, 3, device=device)], dim=-1)
        object_feat_padded  = torch.cat([object_feat, torch.zeros(num_envs, 1, device=device)], dim=-1)


        #  joint_feats: list of [B, 11] → stack to [B, 6, 11]
        joint_feats = torch.stack([joint1_feat, joint2_feat, joint3_feat, joint4_feat, joint5_feat, joint6_feat], dim=1)

        # base + joints + ee + object → [B, 9, 11]
        all_nodes = torch.cat([
            base_feat_padded.unsqueeze(1),  # [B, 1, 11]
            joint_feats,                    # [B, 6, 11]
            ee_feat_padded.unsqueeze(1),    # [B, 1, 11]
            object_feat_padded.unsqueeze(1) # [B, 1, 11]
        ], dim=1)  # shape: [B, 9, 11]

        #  [1,0,0,0], [0,1,0,0], ...
        type_tensor = torch.tensor([
            [1,0,0,0],      # base
            [0,1,0,0],      # joint1
            [0,1,0,0],      # joint2
            [0,1,0,0],      # ...
            [0,1,0,0],
            [0,1,0,0],
            [0,1,0,0],
            [0,0,1,0],      # ee
            [0,0,0,1],      # object
        ], dtype=torch.float32, device=device).unsqueeze(0).expand(num_env, -1, -1)  # [B, 9, 4]

        #  node_feat: [B, 9, 15]
        node_features = torch.cat([all_nodes, type_tensor], dim=-1)  # [B, 9, 15]
        node_features = node_features.reshape(-1, 15)  # [B * 9, 15]

        # Construct pose table => for edge construction purpose
        # base pose: [0,0,0,0,0,0,1]
        base_pose = torch.tensor([0., 0., 0., 0., 0., 0., 1.], device=device).view(1, 1, 7).expand(num_env, 1, 7)

        # Extract out joints, ee, object pose 
        joint_pose = joint_feats[:, :, :7]               # [B, 6, 7]
        ee_pose = ee_feat[:, :7].unsqueeze(1)            # [B, 1, 7]
        object_pose = object_feat[:, :7].unsqueeze(1)    # [B, 1, 7]

        pose_table = torch.cat([base_pose, joint_pose, ee_pose, object_pose], dim=1)  # [B, 9, 7]
        pose_table = pose_table.reshape(-1, 7)  # [B*9, 7]

        # --- batch vector ---
        batch = torch.arange(num_env, device=device).repeat_interleave(9)  # [B*9]

        # Build up the edges
        num_nodes_per_graph = 9
        local_edges = [(0, j) for j in range(1, 7)]  # base→joints
        local_edges += [(j, j+1) for j in range(1, 6)]  # joint chain
        local_edges.append((6, 7))  # last joint→ee
        local_edges.append((7, 8))  # ee→object
        local_edges += [(dst, src) for (src, dst) in local_edges]  # reverse edges

        # The non-vectorized version of constructing edge_index (for reference)
        # edge_index = []
        # for b in range(num_env):
        #     offset = b * num_nodes_per_graph
        #     for src, dst in local_edges:
        #         edge_index.append([offset + src, offset + dst])
        # edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()

        # Indices of the nodes attached to each edge
        num_edges_per_graph = len(local_edges)
        local_edges_tensor = torch.tensor(local_edges, dtype=torch.long, device=device)  # [E, 2]
        src_local, dst_local = local_edges_tensor[:, 0], local_edges_tensor[:, 1]  # [E]

        batch_offsets = torch.arange(num_env, device=device) * num_nodes_per_graph  # [B]
        batch_offsets = batch_offsets.view(-1, 1).expand(-1, num_edges_per_graph)   # [B, E]

        # src & dst of nodes in node_features (Bx9, 15)
        src_all = src_local.view(1, -1) + batch_offsets  # [B, E]
        dst_all = dst_local.view(1, -1) + batch_offsets  # [B, E]

        #  [2, B * E]
        edge_index = torch.stack([
            src_all.reshape(-1),  # [B * E]
            dst_all.reshape(-1)   # [B * E]
        ], dim=0)  # [2, B * E]

        # Calculate the relative pose for each edge & attach it to the edge attributes
        src, dst = edge_index
        pos_src = pose_table[src, :3]
        pos_dst = pose_table[dst, :3]
        rel_pos = pos_dst - pos_src  # [E, 3]

        quat_src = pose_table[src, 3:]
        quat_dst = pose_table[dst, 3:]
        q_src_inv = quat_inverse(quat_src)
        rel_quat = quat_mul(quat_dst, q_src_inv)  # [E, 4]

        edge_attr = torch.cat([rel_pos, rel_quat], dim=-1)  # [E, 7]

        return node_features, edge_index, edge_attr, batch


