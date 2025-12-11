import torch
from torch import nn

def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=gain)

def transitivity_loss(M):
    """
    计算 M 的传递性损失，鼓励 M 满足传递性
    loss = mean( (M @ M)_ik - M_ik )^2
    其中 (M @ M)_ik = 1 - exp( sum_j log(1 - M_ij * M_jk) )
    """
    # 1. 构造 M_ij * M_jk
    # (B, N, N, N)
    term = M.unsqueeze(3) * M.unsqueeze(1)
    
    # 2. Log (1 - P)
    eps = 1e-6
    # clamp 保证 log 输入为正
    log_term = torch.log(torch.clamp(1 - term, min=eps))
    
    # 3. Sum over j (dim=2)
    sum_log = torch.sum(log_term, dim=2)
    
    # 4. Exp and 1 - ...
    M2 = 1 - torch.exp(sum_log)

    # 5. 计算 loss
    result = ((M2 - M) ** 2).mean()
    
    return result

def binarization_loss(M):
    """
    计算 M 的二值化损失，鼓励 M 的元素接近 0 或 1
    loss = mean( M_ij * (1 - M_ij) )
    """
    loss = M * (1 - M)
    return loss.mean()

def symmetry_loss(M):
    """
    计算 M 的对称性损失，鼓励 M 满足对称性
    loss = mean( (M_ij - M_ji)^2 )
    """
    loss = (M - M.transpose(-1, -2)) ** 2
    return loss.mean()

def degree_constraint_loss(M, min_degree, max_degree):
    """
    计算 M 的出度限制损失，鼓励每个节点的出度在指定范围内
    loss = mean( ReLU(out_degree - max_degree) + ReLU(min_degree - out_degree) )
    其中 out_degree_i = sum_j M_ij
    """
    out_degree = M.sum(dim=-1)  # 计算每个节点的出度

    loss_max = torch.relu(out_degree - max_degree)
    loss_min = torch.relu(min_degree - out_degree)

    loss = loss_max + loss_min
    return loss.mean()