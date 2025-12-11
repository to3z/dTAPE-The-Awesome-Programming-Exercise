import torch as th
import numpy as np
from types import SimpleNamespace as SN
from .segment_tree import SumSegmentTree, MinSegmentTree
import random
class EpisodeBatch:
    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,
                 preprocess=None,
                 device="cpu"):
        self.scheme = scheme.copy()
        self.groups = groups
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.preprocess = {} if preprocess is None else preprocess
        self.device = device

        if data is not None:
            self.data = data
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]

                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)

                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]

        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })

        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)

            if isinstance(vshape, int):
                vshape = (vshape,)

            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape

            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            else:
                self.data.transition_data[field_key] = th.zeros((batch_size, max_seq_length, *shape), dtype=dtype, device=self.device)

    def extend(self, scheme, groups=None):
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        slices = self._parse_slices((bs, ts))
        for k, v in data.items():
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            else:
                raise KeyError("{} not found in transition or episode data".format(k))

            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])

            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]
                for transform in self.preprocess[k][1]:
                    v = transform.transform(v)
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            new_data = self._new_data_sn()
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))

            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data, device=self.device)
            return ret
        else:
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]

            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)

            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1)//_range[2]

    def _new_data_sn(self):
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
            or isinstance(items, int)  # int i
            or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
            ):
            items = (items, slice(None))

        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")

        for item in items:
            #TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item+1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        self.buffer_size = buffer_size  # same as self.batch_size but more explicit
        self.buffer_index = 0
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def uni_sample(self, batch_size):
        return self.sample(batch_size)

    def sample_latest(self, batch_size):
        assert self.can_sample(batch_size)
        if self.buffer_index - batch_size < 0:
            #Uniform sampling
            return self.uni_sample(batch_size)
        else:
            # Return the latest
            return self[self.buffer_index - batch_size : self.buffer_index]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())


# Adapted from the OpenAI Baseline implementations (https://github.com/openai/baselines)
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, alpha, beta, t_max, preprocess=None, device="cpu"):
        super(PrioritizedReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length,
                                                      preprocess=preprocess, device="cpu")
        self.alpha = alpha
        self.beta_original = beta
        self.beta = beta
        self.beta_increment = (1.0 - beta) / t_max
        self.max_priority = 1.0

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)

    def insert_episode_batch(self, ep_batch):
        # TODO: convert batch/episode to idx?
        pre_idx = self.buffer_index
        super().insert_episode_batch(ep_batch)
        idx = self.buffer_index
        if idx >= pre_idx:
            for i in range(idx - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
        else:
            for i in range(self.buffer_size - pre_idx):
                self._it_sum[pre_idx + i] = self.max_priority ** self.alpha
                self._it_min[pre_idx + i] = self.max_priority ** self.alpha
            for i in range(self.buffer_index):
                self._it_sum[i] = self.max_priority ** self.alpha
                self._it_min[i] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self.episodes_in_buffer - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, t):
        assert self.can_sample(batch_size)
        self.beta = self.beta_original + (t * self.beta_increment)

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.episodes_in_buffer) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.episodes_in_buffer) ** (-self.beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)

        return self[idxes], idxes, weights

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self.episodes_in_buffer
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class DualReplayBuffer(EpisodeBatch):
    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(DualReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess, device=device)
        # lose buffer index: 0 ~ α*n
        # win buffer index: α*n ~ n-1
        self.win_ratio = 0.5  # 默认胜负比例
        self.win_threshold = 10.0  # 回报阈值，超过为胜利
        self.nxt_lose_idx = 0
        self.nxt_win_idx = buffer_size - 1
        self.n_lose = 0
        self.n_win = 0
        self.buffer_size = buffer_size
        self.lbound = int(self.win_ratio * buffer_size)  # 负样本个数的下限
        self.rbound = buffer_size  # 负样本个数的上限

    def insert_episode_batch(self, ep_batch):
        # # 计算每个 episode 的总回报，根据是否大于阈值来划分胜负
        # episode_returns = ep_batch["reward"].sum(dim=1) # (bs, 1)
        # win_indices = (episode_returns > self.win_threshold).nonzero(as_tuple=True)[0].cpu().tolist()
        # lose_indices = (episode_returns <= self.win_threshold).nonzero(as_tuple=True)[0].cpu().tolist()

        # 根据 battle_won 字段划分胜负
        battle_won = ep_batch["battle_won"].squeeze(-1)
        win_indices = (battle_won == 1).nonzero(as_tuple=True)[0].cpu().tolist()
        lose_indices = (battle_won == 0).nonzero(as_tuple=True)[0].cpu().tolist()
        
        if len(win_indices) > 0:
            self.nxt_win_idx = self._ring_insert(ep_batch[win_indices],
                    self.buffer_size - 1, self.lbound,
                    self.nxt_win_idx)
            self.n_win = min(self.n_win + len(win_indices), self.lbound)
            self.rbound = max(self.lbound, self.buffer_size - self.n_win)
            if self.nxt_lose_idx >= self.rbound:
                # 胜样本的增加可能导致当前的负样本插入索引在边界之外，重置负样本插入索引
                self.nxt_lose_idx = 0
        if len(lose_indices) > 0:
            self.nxt_lose_idx = self._ring_insert(ep_batch[lose_indices],
                    0, self.rbound - 1,
                    self.nxt_lose_idx)
            self.n_lose = min(self.n_lose + len(lose_indices), self.rbound)

    def can_sample(self, batch_size):
        return self.n_win + self.n_lose >= batch_size

    def sample(self, batch_size):
        n_win_sample = min(self.n_win, batch_size // 2)
        n_lose_sample = batch_size - n_win_sample
        win_indices = list(range(self.buffer_size - self.n_win, self.buffer_size))
        lose_indices = list(range(self.n_lose))
        sampled_win_indices = random.sample(win_indices, n_win_sample)
        sampled_lose_indices = random.sample(lose_indices, n_lose_sample)
        return self[sampled_win_indices + sampled_lose_indices]

    def _ring_insert(self, ep_batch, start_idx, end_idx, insert_idx):
        """
        在环形缓冲区中插入一个 episode batch
        Args:
            start_idx: 缓冲区起始索引 (包含)
            end_idx: 缓冲区结束索引 (包含)
            insert_idx: 下一个插入位置
        Returns:
            更新后的下一个插入位置
        """
        bs = ep_batch.batch_size
        buffer_left = abs(end_idx - insert_idx) + 1
        if bs <= buffer_left:
            ascending = start_idx < end_idx
            if not ascending:
                start_idx, end_idx = end_idx, start_idx
            buffer_slice = slice(insert_idx, insert_idx + bs) if ascending else \
                           slice(insert_idx - bs + 1, insert_idx + 1)
            self.update(ep_batch.data.transition_data,
                        buffer_slice,
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data, buffer_slice)
            insert_idx = insert_idx + bs if ascending else insert_idx - bs
            if insert_idx > end_idx:  # ascending
                insert_idx = start_idx
            if insert_idx < start_idx:  # descending
                insert_idx = end_idx
        else:
            insert_idx = self._ring_insert(ep_batch[0:buffer_left, :], start_idx, end_idx, insert_idx)
            insert_idx = self._ring_insert(ep_batch[buffer_left:, :], start_idx, end_idx, insert_idx)
        return insert_idx

    def __repr__(self):
        return f"DualReplayBuffer(Win: {self.win_buffer.episodes_in_buffer}, Lose: {self.lose_buffer.episodes_in_buffer})"
    
    def win_rate(self):
        total = self.n_win + self.n_lose
        return self.n_win / total if total > 0 else 0.0