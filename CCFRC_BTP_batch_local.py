# -*- coding: utf-8 -*-
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from typing import List, Tuple, Dict, Any, Optional
from scipy.io.wavfile import read as wav_read

torch.set_grad_enabled(False)

# ===================== 基础工具函数（保持不变） =====================
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

def cartesian_to_spherical(v: np.ndarray) -> Tuple[float, float]:
    x, y, z = v
    r = float(np.linalg.norm(v))
    azi = float(np.arctan2(y, x) * 180.0 / np.pi)
    ele = float(np.arcsin(np.clip(z / (r + 1e-12), -1.0, 1.0)) * 180.0 / np.pi)
    ele = float(np.clip(ele, 0.0, 90.0))
    return azi, ele

def spherical_to_cartesian(azi_deg: float, ele_deg: float) -> np.ndarray:
    azi = np.deg2rad(azi_deg)
    ele = np.deg2rad(ele_deg)
    x = np.cos(azi) * np.cos(ele)
    y = np.sin(azi) * np.cos(ele)
    z = np.sin(ele)
    return np.array([x, y, z], dtype=np.float32)

def slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    v0 = _unit(v0)
    v1 = _unit(v1)
    dot = np.dot(v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot) * t
    v2 = _unit(v1 - v0 * dot)
    return v0 * np.cos(theta) + v2 * np.sin(theta)

def generate_points_between_points(v1: np.ndarray, v2: np.ndarray, step_deg: float = 1.0) -> np.ndarray:
    v1_unit = _unit(v1)
    v2_unit = _unit(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))
    angle_deg = np.rad2deg(angle_rad)
    if angle_deg < step_deg:
        return np.stack([v1_unit, v2_unit], axis=0)
    num_steps = int(angle_deg / step_deg) + 1
    t_values = np.linspace(0.0, 1.0, num_steps, endpoint=True)
    points = [slerp(v1_unit, v2_unit, t) for t in t_values]
    return np.stack(points, axis=0)

def sphere(levels_count: int = 4) -> torch.Tensor:
    h = (5.0 ** 0.5) / 5.0
    r = (2.0 / 5.0) * (5.0 ** 0.5)
    pi = 3.141592654
    pts = torch.zeros((12, 3), dtype=torch.float)
    pts[0, :] = torch.FloatTensor([0, 0, 1])
    pts[11, :] = torch.FloatTensor([0, 0, -1])
    pts[range(1, 6), 0] = r * torch.sin(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(1, 6), 1] = r * torch.cos(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(1, 6), 2] = h
    pts[range(6, 11), 0] = -r * torch.sin(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(6, 11), 1] = -r * torch.cos(2.0 * pi * torch.arange(0, 5) / 5.0)
    pts[range(6, 11), 2] = -h
    trs = torch.zeros((20, 3), dtype=torch.long)
    trs[0, :] = torch.LongTensor([0, 2, 1])
    trs[1, :] = torch.LongTensor([0, 3, 2])
    trs[2, :] = torch.LongTensor([0, 4, 3])
    trs[3, :] = torch.LongTensor([0, 5, 4])
    trs[4, :] = torch.LongTensor([0, 1, 5])
    trs[5, :] = torch.LongTensor([9, 1, 2])
    trs[6, :] = torch.LongTensor([10, 2, 3])
    trs[7, :] = torch.LongTensor([6, 3, 4])
    trs[8, :] = torch.LongTensor([7, 4, 5])
    trs[9, :] = torch.LongTensor([8, 5, 1])
    trs[10, :] = torch.LongTensor([4, 7, 6])
    trs[11, :] = torch.LongTensor([5, 8, 7])
    trs[12, :] = torch.LongTensor([1, 9, 8])
    trs[13, :] = torch.LongTensor([2, 10, 9])
    trs[14, :] = torch.LongTensor([3, 6, 10])
    trs[15, :] = torch.LongTensor([11, 6, 7])
    trs[16, :] = torch.LongTensor([11, 7, 8])
    trs[17, :] = torch.LongTensor([11, 8, 9])
    trs[18, :] = torch.LongTensor([11, 9, 10])
    trs[19, :] = torch.LongTensor([11, 10, 6])
    for _ in range(0, levels_count):
        trs_count = trs.shape[0]
        subtrs_count = trs_count * 4
        subtrs = torch.zeros((subtrs_count, 6), dtype=torch.long)
        subtrs[0 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 0]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[0 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 0]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 1]
        subtrs[1 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 0]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[2 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 2]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 0] = trs[:, 0]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 1] = trs[:, 1]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 2] = trs[:, 1]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 3] = trs[:, 2]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 4] = trs[:, 2]
        subtrs[3 * trs_count + torch.arange(0, trs_count), 5] = trs[:, 0]
        subtrs_flatten = torch.cat((subtrs[:, [0, 1]], subtrs[:, [2, 3]], subtrs[:, [4, 5]]), axis=0)
        subtrs_sorted, _ = torch.sort(subtrs_flatten, axis=1)
        index_max = torch.max(subtrs_sorted)
        subtrs_scalar = subtrs_sorted[:, 0] * (index_max + 1) + subtrs_sorted[:, 1]
        unique_scalar, unique_indices = torch.unique(subtrs_scalar, return_inverse=True)
        unique_values = torch.zeros((unique_scalar.shape[0], 2), dtype=unique_scalar.dtype)
        unique_values[:, 0] = torch.div(unique_scalar, index_max + 1, rounding_mode="floor")
        unique_values[:, 1] = unique_scalar - unique_values[:, 0] * (index_max + 1)
        trs = torch.transpose(torch.reshape(unique_indices, (3, -1)), 0, 1)
        pts = pts[unique_values[:, 0], :] + pts[unique_values[:, 1], :]
        pts /= torch.repeat_interleave(torch.unsqueeze(torch.sum(pts **2, axis=1)** 0.5, 1), 3, 1)
    return pts

# 球面网格缓存
SPHERE_CACHE: Dict[int, np.ndarray] = {}
def get_sphere_np(level: int) -> np.ndarray:
    arr = SPHERE_CACHE.get(level)
    if arr is None:
        arr = sphere(levels_count=level).numpy().astype(np.float32)
        SPHERE_CACHE[level] = arr
    return arr

# -------------- SRP-PHAT 核心计算（保持不变） --------------
def doas2taus(doas: torch.Tensor, mics: torch.Tensor, fs: float, c: float = 343.0) -> torch.Tensor:
    return (fs / c) * torch.matmul(doas.to(mics.device), mics.transpose(0, 1))

def steering(taus: torch.Tensor, n_fft: int) -> torch.Tensor:
    pi = 3.141592653589793
    frame_size = int((n_fft - 1) * 2)
    omegas = 2 * pi * torch.arange(0, n_fft, device=taus.device) / frame_size
    omegas = omegas.repeat(taus.shape + (1,))
    taus = taus.unsqueeze(len(taus.shape)).repeat((1,) * len(taus.shape) + (n_fft,))
    a_re = torch.cos(-omegas * taus)
    a_im = torch.sin(-omegas * taus)
    a = torch.stack((a_re, a_im), len(a_re.shape))
    a = a.transpose(len(a.shape) - 3, len(a.shape) - 1).transpose(len(a.shape) - 3, len(a.shape) - 2)
    return a

class Covariance(torch.nn.Module):
    def __init__(self, average: bool = True):
        super().__init__(); self.average = average
    def forward(self, Xs: torch.Tensor) -> torch.Tensor:
        return Covariance.cov(Xs=Xs, average=self.average)
    def cov(Xs: torch.Tensor, average: bool = True) -> torch.Tensor:
        n_mics = Xs.shape[4]
        Xs_re = Xs[..., 0, :].unsqueeze(4); Xs_im = Xs[..., 1, :].unsqueeze(4)
        Rxx_re = torch.matmul(Xs_re, Xs_re.transpose(3, 4)) + torch.matmul(Xs_im, Xs_im.transpose(3, 4))
        Rxx_im = torch.matmul(Xs_re, Xs_im.transpose(3, 4)) - torch.matmul(Xs_im, Xs_re.transpose(3, 4))
        idx = torch.triu_indices(n_mics, n_mics)
        XXs_re = Rxx_re[..., idx[0], idx[1]]; XXs_im = Rxx_im[..., idx[0], idx[1]]
        XXs = torch.stack((XXs_re, XXs_im), 3)
        if average:
            n_time_frames = XXs.shape[1]
            XXs = torch.mean(XXs, 1, keepdim=True); XXs = XXs.repeat(1, n_time_frames, 1, 1, 1)
        return XXs

class STFT(torch.nn.Module):
    def __init__(self, sample_rate: int, win_length_ms: int = 20, hop_length_ms: int = 10,
                 n_fft: int = 1024, window_fn=torch.hamming_window, normalized_stft: bool = False,
                 center: bool = True, pad_mode: str = "constant", onesided: bool = True):
        super().__init__()
        self.sample_rate = sample_rate
        self.win_length = int(round((sample_rate / 1000.0) * win_length_ms))
        self.hop_length = int(round((sample_rate / 1000.0) * hop_length_ms))
        self.n_fft = n_fft; self.normalized_stft = normalized_stft
        self.center = center; self.pad_mode = pad_mode; self.onesided = onesided
        self.window = window_fn(self.win_length)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        or_shape = x.shape
        if len(or_shape) == 3:
            x = x.transpose(1, 2); x = x.reshape(or_shape[0]*or_shape[2], or_shape[1])
        stft = torch.stft(x, self.n_fft, self.hop_length, self.win_length,
                          self.window.to(x.device), self.center, self.pad_mode,
                          self.normalized_stft, self.onesided, return_complex=True)
        stft = torch.view_as_real(stft)
        if len(or_shape) == 3:
            stft = stft.reshape(or_shape[0], or_shape[2], stft.shape[1], stft.shape[2], stft.shape[3])
            stft = stft.permute(0, 3, 2, 4, 1)
        else:
            stft = stft.transpose(2, 1)
        return stft

# -------------- 条带掩码生成（仅Stage1使用，保持不变） --------------
def make_elevation_strips_union_masker_variable(centers: List[float], half_degs: List[float]):
    centers = list(map(float, centers))
    half_degs = list(map(float, half_degs))
    assert len(centers) == len(half_degs)
    def masker(grid_np: np.ndarray) -> np.ndarray:
        z = np.clip(grid_np[:, 2], -1.0, 1.0)
        ele = np.rad2deg(np.arcsin(z))
        keep = np.zeros_like(ele, dtype=bool)
        for c, h in zip(centers, half_degs):
            lo, hi = max(0.0, c - h), min(90.0, c + h)
            keep |= (ele >= lo) & (ele <= hi)
        keep &= (ele >= 0.0)
        return grid_np[keep]
    return masker

def filter_by_cap_union(cand_dirs: np.ndarray, center_dirs: np.ndarray,
                        half_angle_deg: float, restrict_upper: bool = True) -> np.ndarray:
    C = _unit(np.asarray(cand_dirs, dtype=np.float32))
    U = _unit(np.asarray(center_dirs, dtype=np.float32))
    cos_thr = np.cos(np.deg2rad(half_angle_deg))
    mask = (C @ U.T) >= cos_thr
    keep = mask.any(axis=1)
    if restrict_upper:
        keep &= (C[:, 2] >= 0)
    return C[keep]

# -------------- SRP-PHAT 类（核心算法，保持不变） --------------
class SrpPhat(torch.nn.Module):
    def __init__(self, mics: torch.Tensor, sample_rate: int = 50000, speed_sound: float = 343.0, eps: float = 1e-20):
        super().__init__()
        self.mics = mics
        self.sample_rate = sample_rate
        self.speed_sound = speed_sound
        self.eps = eps

    def _precompute_XXs(self, mic_signals_np: np.ndarray, n_fft: int = 1024,
                        win_ms: int = 20, hop_ms: int = 10) -> torch.Tensor:
        mic_signals = torch.from_numpy(mic_signals_np.astype(np.float32)).float().unsqueeze(0).transpose(1, 2)
        stft = STFT(sample_rate=self.sample_rate, n_fft=n_fft, win_length_ms=win_ms, hop_length_ms=hop_ms)
        cov = Covariance()
        with torch.no_grad():
            Xs = stft(mic_signals)
            XXs = cov(Xs)
        return XXs

    @staticmethod
    def srp_score_map(XXs: torch.Tensor, As: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
        As = As.to(XXs.device)
        n_mics = As.shape[3]
        idx = torch.triu_indices(n_mics, n_mics)
        As_1_re = As[:, :, 0, idx[0, :]]; As_1_im = As[:, :, 1, idx[0, :]]
        As_2_re = As[:, :, 0, idx[1, :]]; As_2_im = As[:, :, 1, idx[1, :]]
        Ws_re = (As_1_re * As_2_re + As_1_im * As_2_im).reshape(As.shape[0], -1)
        Ws_im = (As_1_re * As_2_im - As_1_im * As_2_re).reshape(As.shape[0], -1)
        XXs_re = XXs[:, :, :, 0, :].reshape(XXs.shape[0], XXs.shape[1], -1)
        XXs_im = XXs[:, :, :, 1, :].reshape(XXs.shape[0], XXs.shape[1], -1)
        XXs_abs = torch.sqrt(XXs_re **2 + XXs_im** 2) + eps
        XXs_re_norm = XXs_re / XXs_abs; XXs_im_norm = XXs_im / XXs_abs
        Ys_A = torch.matmul(XXs_re_norm, Ws_re.transpose(0, 1))
        Ys_B = torch.matmul(XXs_im_norm, Ws_im.transpose(0, 1))
        Ys = Ys_A - Ys_B
        return Ys.mean(dim=1)

    def _score_dirs(self, XXs: torch.Tensor, dirs_np: np.ndarray) -> np.ndarray:
        directions = torch.from_numpy(dirs_np.astype(np.float32)).float()
        taus = doas2taus(directions, mics=self.mics, fs=self.sample_rate, c=self.speed_sound)
        n_fft = XXs.shape[2]
        As = steering(taus, n_fft)
        with torch.no_grad():
            scores = SrpPhat.srp_score_map(XXs=XXs, As=As, eps=self.eps)[0].detach().cpu().numpy()
        return scores

    def coarse_to_fine_search(self, mic_signals_np: Optional[np.ndarray], candidate_grid_np: np.ndarray,
                              max_level: int = 2, topN: int = 8, region_masker=None,
                              XXs: Optional[torch.Tensor] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if XXs is None:
            if mic_signals_np is None:
                raise ValueError("Either XXs or mic_signals_np must be provided.")
            XXs = self._precompute_XXs(mic_signals_np)
        if candidate_grid_np is None or candidate_grid_np.shape[0] == 0:
            candidate_grid_np = sphere(levels_count=1).numpy().astype(np.float32)
            if region_masker is not None and candidate_grid_np.size:
                candidate_grid_np = region_masker(candidate_grid_np)
        current_grid = candidate_grid_np.astype(np.float32)
        level = 0; best_direction = None; top2_directions = None
        angle_ranges = [8, 5, 2, 3]; target_counts = [20, 100, 100, 40]
        while level < max_level:
            directions = torch.from_numpy(current_grid).float()
            if directions.shape[0] == 0: break
            taus = doas2taus(directions, mics=self.mics, fs=self.sample_rate, c=self.speed_sound)
            n_fft = XXs.shape[2]
            As = steering(taus, n_fft)
            srp_scores = SrpPhat.srp_score_map(XXs=XXs, As=As, eps=self.eps)[0]
            N = int(srp_scores.numel())
            if N == 0: break
            k = min(int(topN), N)
            if k <= 0: break
            
            top_indices = torch.topk(srp_scores, k=min(2, N), largest=True).indices
            top_directions = current_grid[top_indices.cpu().numpy()]
            best_direction = top_directions[0] if len(top_directions) > 0 else None
            
            if len(top_directions) >= 2:
                top2_directions = top_directions[:2]
            else:
                if best_direction is not None:
                    top2_directions = np.repeat(best_direction[np.newaxis, :], 2, axis=0)
                else:
                    top2_directions = None
            
            if level == max_level - 1: break
            LEVEL_SCHEDULE = [3, 5, 6, 6]
            refine_level = LEVEL_SCHEDULE[level] if level < len(LEVEL_SCHEDULE) else LEVEL_SCHEDULE[-1]
            cand = get_sphere_np(refine_level)
            angle_base = angle_ranges[level] if level < len(angle_ranges) else angle_ranges[-1]
            angle_range = float(angle_base)
            tc = target_counts[level] if level < len(target_counts) else target_counts[-1]
            refined = cand
            for _ in range(5):
                refined = filter_by_cap_union(cand_dirs=refined, center_dirs=top_directions,
                                              half_angle_deg=angle_range, restrict_upper=True)
                if region_masker is not None and refined.size:
                    refined = region_masker(refined)
                if 0.8*tc <= len(refined) <= 1.2*tc: break
                if len(refined) > 1.2*tc: angle_range *= 0.8
                else:                    angle_range *= 1.25
            if len(refined) == 0:
                refined = filter_by_cap_union(cand_dirs=cand, center_dirs=top_directions,
                                              half_angle_deg=angle_base*1.5, restrict_upper=True)
                if region_masker is not None and refined.size:
                    refined = region_masker(refined)
                if len(refined) == 0: break
            current_grid = refined.astype(np.float32); level += 1
        
        return best_direction, top2_directions

    def stage2_between_points_search(self,
                                    XXs: torch.Tensor,
                                    top2_dirs: np.ndarray,
                                    step_deg: float = 1.0,
                                    quad_interp: bool = True
                                    ) -> Tuple[float, float, Dict[str, Any]]:
        if top2_dirs is None or len(top2_dirs) < 2:
            raise ValueError("需要至少两个方向点用于Stage2搜索")
        
        t0 = time.time()
        candidate_points = generate_points_between_points(
            top2_dirs[0], top2_dirs[1], step_deg=step_deg
        )
        if len(candidate_points) < 2:
            raise RuntimeError("在两个点之间无法生成足够的候选点")
        
        scores = self._score_dirs(XXs, candidate_points)
        t_eval = time.time() - t0
        
        idx = int(np.argmax(scores))
        best_point = candidate_points[idx]
        az_star, el_star = cartesian_to_spherical(best_point)
        
        if quad_interp and 0 < idx < len(scores) - 1:
            azi1, ele1 = cartesian_to_spherical(candidate_points[idx-1])
            azi2, ele2 = az_star, el_star
            azi3, ele3 = cartesian_to_spherical(candidate_points[idx+1])
            
            angle1 = np.rad2deg(np.arccos(np.clip(np.dot(
                _unit(candidate_points[idx-1]), _unit(candidate_points[idx])), -1.0, 1.0)))
            angle3 = np.rad2deg(np.arccos(np.clip(np.dot(
                _unit(candidate_points[idx]), _unit(candidate_points[idx+1])), -1.0, 1.0)))
            
            x1, x2, x3 = ele1, ele2, ele3
            y1, y2, y3 = scores[idx-1], scores[idx], scores[idx+1]
            denom = (y3 - 2*y2 + y1)
            if abs(denom) > 1e-12:
                delta = (y3 - y1) / (2.0 * denom)
                el_star = float(np.clip(x2 - delta * (x3 - x2), 0.0, 90.0))
            
            x1, x2, x3 = azi1, azi2, azi3
            denom = (y3 - 2*y2 + y1)
            if abs(denom) > 1e-12:
                delta = (y3 - y1) / (2.0 * denom)
                az_star = float(x2 - delta * (x3 - x2))
                az_star = (az_star + 180) % 360 - 180
        
        az1, el1 = cartesian_to_spherical(top2_dirs[0])
        az2, el2 = cartesian_to_spherical(top2_dirs[1])
        angle_between_deg = np.rad2deg(np.arccos(np.clip(
            np.dot(_unit(top2_dirs[0]), _unit(top2_dirs[1])), -1.0, 1.0)))
        
        return az_star, el_star, {
            "eval_time": t_eval,
            "num_candidates": len(candidate_points),
            "step_deg": step_deg,
            "angle_between_deg": angle_between_deg,
            "point1_az_deg": az1,
            "point1_el_deg": el1,
            "point2_az_deg": az2,
            "point2_el_deg": el2
        }

    def two_stage_search(self,
                         mic_signals_np: np.ndarray,
                         strip_centers_stage1: Tuple[float, ...] = (9, 27, 45, 63, 81),
                         fixed_half_deg_stage1: float = 5.0,
                         hmin_deg_stage1: float = 2.0, hmax_deg_stage1: float = 12.0,
                         level_schedule_stage0: Tuple[int, int] = (2,4),
                         topN_stage0: Tuple[int, int] = (10, 6),
                         step_stage2: float = 1.0,
                         quad_interp_stage2: bool = True,
                         n_fft: int = 1024, win_ms: int = 20, hop_ms: int = 10
                         ) -> Tuple[float, float, Dict[str, Any]]:
        t_total = time.time()

        t0 = time.time()
        XXs = self._precompute_XXs(mic_signals_np, n_fft=n_fft, win_ms=win_ms, hop_ms=hop_ms)
        t_pre = time.time() - t0

        t0 = time.time()
        fixed_half_deg_clipped = np.clip(fixed_half_deg_stage1, hmin_deg_stage1, hmax_deg_stage1)
        half_list_stage1 = [fixed_half_deg_clipped] * len(strip_centers_stage1)
        region_union_stage1 = make_elevation_strips_union_masker_variable(
            list(strip_centers_stage1), half_list_stage1
        )
        
        cand0 = None
        for lvl in (2, 3):
            base = get_sphere_np(lvl)
            tmp = region_union_stage1(base)
            if tmp.size > 0:
                cand0 = tmp; break
        if cand0 is None or cand0.size == 0:
            wide_half_deg = np.clip(fixed_half_deg_clipped + 2.0, hmin_deg_stage1, hmax_deg_stage1)
            region_wide = make_elevation_strips_union_masker_variable(
                list(strip_centers_stage1), [wide_half_deg]*len(strip_centers_stage1)
            )
            cand0 = region_wide(get_sphere_np(6))
            if cand0.size == 0:
                raise RuntimeError("Stage1 候选为空，请调整条带配置.")
        
        best_dir0, top2_dirs = self.coarse_to_fine_search(
            mic_signals_np=None, candidate_grid_np=cand0, max_level=len(level_schedule_stage0),
            topN=topN_stage0[0], region_masker=None, XXs=XXs
        )
        if best_dir0 is None or top2_dirs is None:
            raise RuntimeError("Stage1 未获得有效方向。")
        az_stage1, el_stage1 = cartesian_to_spherical(best_dir0)
        az1, el1 = cartesian_to_spherical(top2_dirs[0])
        az2, el2 = cartesian_to_spherical(top2_dirs[1])
        t_stage1 = time.time() - t0

        t0 = time.time()
        az_star, el_star, ls_diag = self.stage2_between_points_search(
            XXs=XXs,
            top2_dirs=top2_dirs,
            step_deg=step_stage2,
            quad_interp=quad_interp_stage2
        )
        t_stage2 = time.time() - t0

        diag = {
            "stage1_strip_centers": strip_centers_stage1,
            "stage1_strip_half_degs": half_list_stage1,
            "stage1_top1_az_deg": az1,
            "stage1_top1_el_deg": el1,
            "stage1_top2_az_deg": az2,
            "stage1_top2_el_deg": el2,
            "stage1_angle_between_deg": ls_diag["angle_between_deg"],
            "stage2_step_deg": step_stage2,
            "timing_sec": {
                "precompute_xxs": t_pre,
                "stage1": t_stage1,
                "stage2": t_stage2,
                "stage2_eval_only": ls_diag["eval_time"],
                "total": time.time() - t_total
            },
            "params": {
                "n_fft": n_fft, "win_ms": win_ms, "hop_ms": hop_ms,
                "quad_interp_stage2": quad_interp_stage2
            },
            "stage2_num_candidates": ls_diag["num_candidates"]
        }
        return az_star, el_star, diag

# ===================== 批量处理核心函数 =====================
def init_microphone_array() -> torch.Tensor:
    """初始化8元圆阵麦克风配置（仅执行一次）"""
    mics = torch.zeros((8, 3))
    radius = 88.75 / 1000 / 2  # 半径（米）
    angles = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315], dtype=torch.float32) * np.pi / 180
    for i in range(8):
        mics[i, :] = torch.tensor([
            radius * torch.cos(angles[i]),
            radius * torch.sin(angles[i]),
            0.0  # z轴为0，平面阵列
        ])
    return mics

def process_single_wav(wav_file_path: str, mics: torch.Tensor) -> Tuple[bool, str, Optional[float], Optional[float], Optional[Dict[str, Any]]]:
    """
    处理单个WAV文件，返回处理结果
    返回值：(处理成功标志, 文件名, 方位角, 俯仰角, 诊断信息)
    """
    filename = os.path.basename(wav_file_path)
    try:
        # 1. 读取WAV文件
        fs, wav_data = wav_read(wav_file_path)
        print(f"\n{'='*120}")
        print(f"正在处理文件: {filename}")
        print(f"文件路径: {wav_file_path}")
        print(f"音频参数: 采样率={fs}Hz, 原始形状={wav_data.shape}")

        # 2. 音频数据格式处理（确保[8麦克风, 采样数]）
        if len(wav_data.shape) == 1:
            raise ValueError("❌ 单通道音频不支持，请提供8通道音频")
        elif wav_data.shape[1] != 8:
            raise ValueError(f"❌ 音频通道数({wav_data.shape[1]})与麦克风数量(8)不匹配")
        mic_signals_np = wav_data.T.astype(np.float32)  # 转置为[8, T]
        print(f"处理后音频形状: {mic_signals_np.shape} (麦克风数, 采样点数)")

        # 3. 初始化SRP-PHAT（按当前文件采样率）
        srpphat = SrpPhat(mics=mics, sample_rate=fs, speed_sound=343.0)

        # 4. 运行两阶段定位
        start_time = time.time()
        az, el, diag = srpphat.two_stage_search(
            mic_signals_np=mic_signals_np,
            strip_centers_stage1=(7.5, 22.5, 37.5, 52.5, 67.5, 82.5),  # 俯仰角条带中心
            fixed_half_deg_stage1=3.0,  # 条带半宽
            hmin_deg_stage1=2.0, 
            hmax_deg_stage1=12.0,
            level_schedule_stage0=(3, 4, 5),  # 粗搜索网格层级
            topN_stage0=(2, 2),  # 每级保留的候选点数
            step_stage2=1.0,  # 1°步进搜索
            quad_interp_stage2=True,  # 启用二次插值优化
            n_fft=1024, 
            win_ms=20, 
            hop_ms=10
        )
        total_time = time.time() - start_time

        # 5. 输出当前文件结果
        print("\n✅ 定位成功！")
        print(f"估计方位角: {az:.2f}°")
        print(f"估计俯仰角: {el:.2f}°")
        print("\nStage1 中间结果:")
        print(f"  最高能量点1: 方位角={diag['stage1_top1_az_deg']:.2f}°, 俯仰角={diag['stage1_top1_el_deg']:.2f}°")
        print(f"  最高能量点2: 方位角={diag['stage1_top2_az_deg']:.2f}°, 俯仰角={diag['stage1_top2_el_deg']:.2f}°")
        print(f"  两点夹角: {diag['stage1_angle_between_deg']:.2f}°")
        print("\n时间统计:")
        print(f"  预处理(STFT+协方差): {diag['timing_sec']['precompute_xxs']:.4f}s")
        print(f"  Stage1粗搜索: {diag['timing_sec']['stage1']:.4f}s")
        print(f"  Stage2精细搜索: {diag['timing_sec']['stage2']:.4f}s")
        print(f"  文件总耗时: {total_time:.4f}s")
        print(f"{'='*120}")

        return True, filename, az, el, diag

    except Exception as e:
        print(f"\n❌ 处理文件 {filename} 失败: {str(e)}")
        print(f"{'='*120}")
        return False, filename, None, None, None

def batch_process_wav_dir(wav_dir: str) -> None:
    """批量处理指定目录下所有WAV文件"""
    # 1. 初始化麦克风阵列（全局唯一）
    mics = init_microphone_array()
    print(f"初始化8元圆阵麦克风配置完成")

    # 2. 获取目录下所有WAV文件（含.WAV/.wav）
    wav_files = []
    for root, _, files in os.walk(wav_dir):  # 支持子目录遍历
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    
    if not wav_files:
        print(f"⚠️ 在目录 {wav_dir} 中未找到任何WAV文件")
        return
    
    # 3. 统计信息初始化
    total_files = len(wav_files)
    success_count = 0
    fail_count = 0
    success_results = []  # 存储成功结果：(filename, az, el)

    # 4. 批量处理
    batch_start_time = time.time()
    print(f"\n📊 开始批量处理：目录={wav_dir}, 总WAV文件数={total_files}")
    for wav_file in wav_files:
        success, filename, az, el, _ = process_single_wav(wav_file, mics)
        if success:
            success_count += 1
            success_results.append((filename, az, el))
        else:
            fail_count += 1

    # 5. 输出批量处理汇总
    batch_total_time = time.time() - batch_start_time
    print(f"\n{'='*80}")
    print(f"📋 批量处理汇总")
    print(f"{'='*80}")
    print(f"总处理文件数: {total_files}")
    print(f"成功处理数: {success_count}")
    print(f"失败处理数: {fail_count}")
    print(f"批量总耗时: {batch_total_time:.4f}s")
    print(f"平均每个文件耗时: {batch_total_time/total_files:.4f}s")
    
    if success_results:
        print(f"\n✅ 成功处理文件详情：")
        for filename, az, el in success_results:
            print(f"  {filename}: 方位角={az:.2f}°, 俯仰角={el:.2f}°")
    print(f"{'='*80}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # --------------------------
    # 配置：修改为你的WAV文件目录
    # --------------------------
    WAV_DIRECTORY = "D:\\SSL\\sslcode\\real world code\\wav"  # 目标目录

    # 启动批量处理
    batch_process_wav_dir(WAV_DIRECTORY)