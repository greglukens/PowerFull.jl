#!/usr/bin/env python3
"""
calcClGR.py
=============================================================================

Python module for computing C_ℓ^GR from pre-computed integrals.

Requires: numpy, h5py
Does NOT require: Julia

Usage:
    from calcClGR import IntegralCollection, ClGRParams, compute_Cl_GR

    I = IntegralCollection("ClGR_integrals.h5")
    nr = len(I.rr)
    params = ClGRParams(
        D = np.array([D(r) for r in I.rr]),
        H = np.array([H(r) for r in I.rr]),
        bg = np.full(nr, 1.5),
        ...
    )
    Cl_map = compute_Cl_GR(I, params, ell=100)  # returns (nr, nR) array

December 2025
Donghui Jeong
=============================================================================
"""

import numpy as np
import h5py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.interpolate import interp1d
import os


def _make_interp(rr: np.ndarray, values: np.ndarray) -> Callable[[float], float]:
    """Create log-interpolation function from arrays."""
    log_rr = np.log10(rr)
    itp = interp1d(log_rr, values, kind='linear',
                   bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda r: float(itp(np.log10(r)))


def _process_param(param, rr_common: Optional[np.ndarray]) -> Callable[[float], float]:
    """Convert array to function if needed."""
    if callable(param):
        return param
    elif isinstance(param, np.ndarray) and rr_common is not None:
        return _make_interp(rr_common, param)
    elif isinstance(param, tuple) and len(param) == 2:
        return _make_interp(param[0], param[1])
    else:
        raise ValueError("Parameter must be callable, array (with rr), or (rr, values) tuple")


@dataclass
class ClGRParams:
    """
    Cosmological parameters for C_ℓ^GR calculation.

    Constructor accepts either:
    - Arrays: with rr grid → auto-creates interpolation function
    - Function: r -> value → uses directly

    Function Attributes (internally stored as callables):
        D: Growth factor D(r)
        H: Conformal Hubble ℋ(r) in units of c/Mpc
        bg: Galaxy bias b_g(r)
        beta: RSD parameter β(r) = f(r)/b_g(r)
        B: ℬ(r) parameter (velocity contribution)
        A: 𝒜(r) parameter (potential contribution)
        Q: Magnification bias 𝒬(r)
        bPhi: Scale-dependent bias b_Φ(r)

    Scalar Attributes:
        f_NL: Primordial non-Gaussianity parameter (default: 0)
        Omm0: Ω_{m,0} at z=0 (default: 0.3)
        H0: H₀ in km/s/Mpc (default: 67.0)

    Examples:
        # Example 1: Arrays with common rr grid
        params = ClGRParams.from_arrays(
            rr=rr_grid,
            D=D_values, H=H_values, bg=bg_values, beta=beta_values,
            B=B_values, A=A_values, Q=Q_values, bPhi=bPhi_values
        )

        # Example 2: Mix of arrays and functions
        params = ClGRParams.from_arrays(
            rr=rr_grid,
            D=D_values,           # array → interpolated
            bg=lambda r: 1.5,     # function → direct
            ...
        )
    """
    # Functions (internally stored as callables)
    D: Callable[[float], float]
    H: Callable[[float], float]
    bg: Callable[[float], float]
    beta: Callable[[float], float]
    B: Callable[[float], float]
    A: Callable[[float], float]
    Q: Callable[[float], float]
    bPhi: Callable[[float], float]

    # Scalars
    f_NL: float = 0.0
    Omm0: float = 0.3
    H0: float = 67.0

    @classmethod
    def from_arrays(cls, rr: Optional[np.ndarray] = None, *,
                    D, H, bg, beta, B, A, Q, bPhi,
                    f_NL: float = 0.0, Omm0: float = 0.3, H0: float = 67.0):
        """Construct from arrays (auto-interpolated) or functions."""
        return cls(
            D=_process_param(D, rr),
            H=_process_param(H, rr),
            bg=_process_param(bg, rr),
            beta=_process_param(beta, rr),
            B=_process_param(B, rr),
            A=_process_param(A, rr),
            Q=_process_param(Q, rr),
            bPhi=_process_param(bPhi, rr),
            f_NL=f_NL, Omm0=Omm0, H0=H0
        )


class IntegralCollection:
    """
    Container for pre-computed integrals.

    Key format: (type, p, j, jp, sub)
    - type: str - Base function or integral type
    - p: int - Power index (-4, -3, -2, -1, 0)
    - j, jp: int - Bessel derivative orders (0, 1, 2)
    - sub: str - Subscript ('r', 'rp', 'r_rp', 'rp_r', 'none')

    Attributes:
        data: Dict mapping keys to 3D arrays [nr, nR, n_ell]
        rr: Radial grid (Mpc/h)
        RR: R = r'/r grid
        ell_values: Multipole ℓ values
    """

    def __init__(self, filename: str):
        self.data: Dict[Tuple[str, int, int, int, str], np.ndarray] = {}
        self.rr: np.ndarray = np.array([])
        self.RR: np.ndarray = np.array([])
        self.ell_values: np.ndarray = np.array([])
        self._load(filename)

    def __getitem__(self, key: Tuple[str, int, int, int, str]) -> np.ndarray:
        if key not in self.data:
            available = list(self.data.keys())[:10]
            raise KeyError(f"Key {key} not found. Available (first 10): {available}")
        return self.data[key]

    def __contains__(self, key: Tuple[str, int, int, int, str]) -> bool:
        return key in self.data

    def keys(self):
        return self.data.keys()

    def _decode_p(self, p_str: str) -> int:
        if p_str.startswith('m'):
            return -int(p_str[1:])
        return int(p_str)

    def _parse_key(self, key: str, is_base: bool) -> Tuple[str, int, int, int, str]:
        parts = key.split('_')
        type_name = parts[0]
        p = self._decode_p(parts[1])
        j = int(parts[2])
        jp = int(parts[3])
        sub = 'none' if is_base else '_'.join(parts[4:])
        return (type_name, p, j, jp, sub)

    def _load(self, filename: str):
        with h5py.File(filename, 'r') as f:
            self.rr = np.array(f['grid/rr'][:], dtype=np.float64)
            self.RR = np.array(f['grid/RR'][:], dtype=np.float64)
            self.ell_values = np.array(f['grid/ell_values'][:])

            if 'base' in f:
                for key in f['base'].keys():
                    parsed = self._parse_key(key, is_base=True)
                    self.data[parsed] = np.array(f['base'][key][:], dtype=np.float64)

            if 'integrated' in f:
                for key in f['integrated'].keys():
                    parsed = self._parse_key(key, is_base=False)
                    self.data[parsed] = np.array(f['integrated'][key][:], dtype=np.float64)

        # Get actual sizes from loaded data
        nr = len(self.rr)
        nR = len(self.RR)
        n_ell = len(self.ell_values)

        print(f"Loaded {len(self.data)} arrays from {filename}")
        print(f"  - r grid: {nr} points, range [{self.rr[0]:.1f}, {self.rr[-1]:.1f}] Mpc/h")
        print(f"  - R grid: {nR} points, range [{self.RR[0]:.3f}, {self.RR[-1]:.3f}]")
        print(f"  - ℓ values: {n_ell}, range [{self.ell_values[0]}, {self.ell_values[-1]}]")

    def show_available_keys(self):
        sorted_keys = sorted(self.data.keys())
        print(f"Available integrals ({len(sorted_keys)} total):")
        for k in sorted_keys:
            print(f"  I['{k[0]}', {k[1]}, {k[2]}, {k[3]}, '{k[4]}']")


def _find_nearest_idx(rr: np.ndarray, r: float) -> Optional[int]:
    """Find nearest index in rr for value r. Returns None if out of range."""
    if r < rr[0] or r > rr[-1]:
        return None
    log_r = np.log10(r)
    log_rr = np.log10(rr)
    idx = np.searchsorted(log_rr, log_r)
    if idx == 0:
        return 0
    elif idx >= len(rr):
        return len(rr) - 1
    else:
        return idx if abs(log_rr[idx] - log_r) < abs(log_rr[idx-1] - log_r) else idx - 1


def _find_ell_idx(ell_values: np.ndarray, ell: int) -> int:
    """Find index of ell in ell_values. Raises error if not found."""
    idx = np.where(ell_values == ell)[0]
    if len(idx) == 0:
        raise ValueError(f"ℓ = {ell} not found. Available: [{ell_values[0]}, {ell_values[-1]}]")
    return idx[0]


def compute_Cl_GR(I: IntegralCollection, params: ClGRParams, ell: int) -> np.ndarray:
    """
    Compute C_ℓ^GR(r,R) from Eq. (C.14) for a given ℓ value.

    Args:
        I: IntegralCollection with all TwoFAST integrals
        params: ClGRParams with physical parameters (arrays on r grid)
        ell: The multipole ℓ value (not index!)

    Returns:
        np.ndarray: C_ℓ^GR values of size (nr, nR)
    """
    ell_idx = _find_ell_idx(I.ell_values, ell)

    # Get actual sizes from data
    nr = len(I.rr)
    nR = len(I.RR)
    result = np.zeros((nr, nR))

    f_NL = params.f_NL
    Omm0 = params.Omm0
    H0 = params.H0
    fNL_prefactor = 1.5 * Omm0 * (H0 / 2997.9)**2

    # Pre-extract integral slices
    w_0_0_0 = I[('w', 0, 0, 0, 'none')][:, :, ell_idx]
    w_0_2_0 = I[('w', 0, 2, 0, 'none')][:, :, ell_idx]
    w_0_0_2 = I[('w', 0, 0, 2, 'none')][:, :, ell_idx]
    w_0_2_2 = I[('w', 0, 2, 2, 'none')][:, :, ell_idx]
    w_m1_0_1 = I[('w', -1, 0, 1, 'none')][:, :, ell_idx]
    w_m1_2_1 = I[('w', -1, 2, 1, 'none')][:, :, ell_idx]
    w_m1_1_0 = I[('w', -1, 1, 0, 'none')][:, :, ell_idx]
    w_m1_1_2 = I[('w', -1, 1, 2, 'none')][:, :, ell_idx]
    w_m2_0_0 = I[('w', -2, 0, 0, 'none')][:, :, ell_idx]
    w_m2_2_0 = I[('w', -2, 2, 0, 'none')][:, :, ell_idx]
    w_m2_0_2 = I[('w', -2, 0, 2, 'none')][:, :, ell_idx]
    w_m2_1_1 = I[('w', -2, 1, 1, 'none')][:, :, ell_idx]
    w_m3_1_0 = I[('w', -3, 1, 0, 'none')][:, :, ell_idx]
    w_m3_0_1 = I[('w', -3, 0, 1, 'none')][:, :, ell_idx]
    w_m4_0_0 = I[('w', -4, 0, 0, 'none')][:, :, ell_idx]

    s_m2_0_0_r = I[('s', -2, 0, 0, 'r')][:, :, ell_idx]
    s_m2_0_2_r = I[('s', -2, 0, 2, 'r')][:, :, ell_idx]
    s_m2_0_0_rp = I[('s', -2, 0, 0, 'rp')][:, :, ell_idx]
    s_m2_2_0_rp = I[('s', -2, 2, 0, 'rp')][:, :, ell_idx]
    s_m3_0_1_r = I[('s', -3, 0, 1, 'r')][:, :, ell_idx]
    s_m3_1_0_rp = I[('s', -3, 1, 0, 'rp')][:, :, ell_idx]
    s_m4_0_0_r = I[('s', -4, 0, 0, 'r')][:, :, ell_idx]
    s_m4_0_0_rp = I[('s', -4, 0, 0, 'rp')][:, :, ell_idx]

    t_m2_0_0_r = I[('t', -2, 0, 0, 'r')][:, :, ell_idx]
    t_m2_0_2_r = I[('t', -2, 0, 2, 'r')][:, :, ell_idx]
    t_m2_0_0_rp = I[('t', -2, 0, 0, 'rp')][:, :, ell_idx]
    t_m2_2_0_rp = I[('t', -2, 2, 0, 'rp')][:, :, ell_idx]
    t_m3_0_1_r = I[('t', -3, 0, 1, 'r')][:, :, ell_idx]
    t_m3_1_0_rp = I[('t', -3, 1, 0, 'rp')][:, :, ell_idx]
    t_m4_0_0_r = I[('t', -4, 0, 0, 'r')][:, :, ell_idx]
    t_m4_0_0_rp = I[('t', -4, 0, 0, 'rp')][:, :, ell_idx]

    l_m2_0_0_r = I[('l', -2, 0, 0, 'r')][:, :, ell_idx]
    l_m2_0_2_r = I[('l', -2, 0, 2, 'r')][:, :, ell_idx]
    l_m2_0_0_rp = I[('l', -2, 0, 0, 'rp')][:, :, ell_idx]
    l_m2_2_0_rp = I[('l', -2, 2, 0, 'rp')][:, :, ell_idx]
    l_m3_0_1_r = I[('l', -3, 0, 1, 'r')][:, :, ell_idx]
    l_m3_1_0_rp = I[('l', -3, 1, 0, 'rp')][:, :, ell_idx]
    l_m4_0_0_r = I[('l', -4, 0, 0, 'r')][:, :, ell_idx]
    l_m4_0_0_rp = I[('l', -4, 0, 0, 'rp')][:, :, ell_idx]

    scrX_r_rp = I[('scrX', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrX_rp_r = I[('scrX', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrY_r_rp = I[('scrY', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrY_rp_r = I[('scrY', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrZ_r_rp = I[('scrZ', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrZ_rp_r = I[('scrZ', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrS_r_rp = I[('scrS', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrT_r_rp = I[('scrT', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrL_r_rp = I[('scrL', -4, 0, 0, 'r_rp')][:, :, ell_idx]

    u_m2_0_0 = I[('u', -2, 0, 0, 'none')][:, :, ell_idx]
    u_m2_2_0 = I[('u', -2, 2, 0, 'none')][:, :, ell_idx]
    u_m2_0_2 = I[('u', -2, 0, 2, 'none')][:, :, ell_idx]
    u_m3_1_0 = I[('u', -3, 1, 0, 'none')][:, :, ell_idx]
    u_m3_0_1 = I[('u', -3, 0, 1, 'none')][:, :, ell_idx]
    u_m4_0_0 = I[('u', -4, 0, 0, 'none')][:, :, ell_idx]

    scrs_m4_r = I[('scrs', -4, 0, 0, 'r')][:, :, ell_idx]
    scrs_m4_rp = I[('scrs', -4, 0, 0, 'rp')][:, :, ell_idx]
    scrt_m4_r = I[('scrt', -4, 0, 0, 'r')][:, :, ell_idx]
    scrt_m4_rp = I[('scrt', -4, 0, 0, 'rp')][:, :, ell_idx]
    scrl_m4_r = I[('scrl', -4, 0, 0, 'r')][:, :, ell_idx]
    scrl_m4_rp = I[('scrl', -4, 0, 0, 'rp')][:, :, ell_idx]

    v_m4_0_0 = I[('v', -4, 0, 0, 'none')][:, :, ell_idx]

    # Loop over all (r, R) grid points
    for iR in range(nR):
        R = I.RR[iR]
        for ir1 in range(nr):
            r1 = I.rr[ir1]
            r2 = R * r1

            # Skip if r2 is out of r grid range
            if r2 < I.rr[0] or r2 > I.rr[-1]:
                result[ir1, iR] = 0.0
                continue

            # Get parameters at r1 and r2 (function calls)
            D1, D2 = params.D(r1), params.D(r2)
            H1, H2 = params.H(r1), params.H(r2)
            bg1, bg2 = params.bg(r1), params.bg(r2)
            beta1, beta2 = params.beta(r1), params.beta(r2)
            B1, B2 = params.B(r1), params.B(r2)
            A1, A2 = params.A(r1), params.A(r2)
            Q1, Q2 = params.Q(r1), params.Q(r2)
            bPhi1, bPhi2 = params.bPhi(r1), params.bPhi(r2)

            prefactor = D1 * D2

            # Term 1
            term1 = bg1 * bg2 * (
                w_0_0_0[ir1, iR] - beta1 * w_0_2_0[ir1, iR]
                - beta2 * w_0_0_2[ir1, iR] + beta1 * beta2 * w_0_2_2[ir1, iR]
            )

            # Term 2
            term2 = (
                bg1 * H2 * B2 * (w_m1_0_1[ir1, iR] - beta1 * w_m1_2_1[ir1, iR])
                + bg2 * H1 * B1 * (w_m1_1_0[ir1, iR] - beta2 * w_m1_1_2[ir1, iR])
            )

            # Term 3
            term3 = bg1 * H2**2 * A2 * (
                w_m2_0_0[ir1, iR] - beta1 * w_m2_2_0[ir1, iR]
                + (H1 / bg1) * B1 * w_m3_1_0[ir1, iR]
            )

            # Term 4
            term4 = bg2 * H1**2 * A1 * (
                w_m2_0_0[ir1, iR] - beta2 * w_m2_0_2[ir1, iR]
                + (H2 / bg2) * B2 * w_m3_0_1[ir1, iR]
            )

            # Term 5
            term5 = bg1 * (B2 / beta2) * (
                s_m2_0_0_rp[ir1, iR] - beta1 * s_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * s_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * s_m4_0_0_rp[ir1, iR]
            )

            # Term 6
            term6 = bg2 * (B1 / beta1) * (
                s_m2_0_0_r[ir1, iR] - beta2 * s_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * s_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * s_m4_0_0_r[ir1, iR]
            )

            # Term 7
            term7 = -2 * bg1 * ((1 - Q2) / r2) * (
                t_m2_0_0_rp[ir1, iR] - beta1 * t_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * t_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * t_m4_0_0_rp[ir1, iR]
            )

            # Term 8
            term8 = -2 * bg2 * ((1 - Q1) / r1) * (
                t_m2_0_0_r[ir1, iR] - beta2 * t_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * t_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * t_m4_0_0_r[ir1, iR]
            )

            # Term 9
            term9 = -2 * bg1 * (1 - Q2) * (
                l_m2_0_0_rp[ir1, iR] - beta1 * l_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * l_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * l_m4_0_0_rp[ir1, iR]
            )

            # Term 10
            term10 = -2 * bg2 * (1 - Q1) * (
                l_m2_0_0_r[ir1, iR] - beta2 * l_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * l_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * l_m4_0_0_r[ir1, iR]
            )

            # Term 11
            term11 = -2 * bg1 * (1 - Q2) * (
                (B1 / (beta1 * r2)) * scrX_r_rp[ir1, iR]
                + (B1 / beta1) * scrY_r_rp[ir1, iR]
                - 2 * ((1 - Q1) / (bg1 * r2)) * scrZ_rp_r[ir1, iR]
            )

            # Term 12
            term12 = -2 * bg2 * (1 - Q1) * (
                (B2 / (beta2 * r1)) * scrX_rp_r[ir1, iR]
                + (B2 / beta2) * scrY_rp_r[ir1, iR]
                - 2 * ((1 - Q2) / (bg2 * r1)) * scrZ_r_rp[ir1, iR]
            )

            # Term 13
            term13 = fNL_prefactor * bg1 * (bPhi2 / D2) * f_NL * (
                u_m2_0_0[ir1, iR] - beta1 * u_m2_2_0[ir1, iR]
                + (H1 / bg1) * B1 * u_m3_1_0[ir1, iR]
                + (H1**2 / bg1) * A1 * u_m4_0_0[ir1, iR]
            )

            # Term 14
            term14 = fNL_prefactor * bg2 * (bPhi1 / D1) * f_NL * (
                u_m2_0_0[ir1, iR] - beta2 * u_m2_0_2[ir1, iR]
                + (H2 / bg2) * B2 * u_m3_0_1[ir1, iR]
                + (H2**2 / bg2) * A2 * u_m4_0_0[ir1, iR]
            )

            # Term 15
            term15 = fNL_prefactor * (bPhi1 / D1) * f_NL * (
                (B2 / (bg2 * beta2)) * scrs_m4_rp[ir1, iR]
                - 2 * (1 - Q2) * ((1 / r2) * scrt_m4_rp[ir1, iR] + scrl_m4_rp[ir1, iR])
            )

            # Term 16
            term16 = fNL_prefactor * (bPhi2 / D2) * f_NL * (
                (B1 / (bg1 * beta1)) * scrs_m4_r[ir1, iR]
                - 2 * (1 - Q1) * ((1 / r1) * scrt_m4_r[ir1, iR] + scrl_m4_r[ir1, iR])
            )

            # Term 17
            term17 = (
                H1 * H2 * B1 * B2 * w_m2_1_1[ir1, iR]
                + H1**2 * H2**2 * A1 * A2 * w_m4_0_0[ir1, iR]
                + (B1 / (bg1 * beta1)) * (B2 / (bg2 * beta2)) * scrS_r_rp[ir1, iR]
            )

            # Term 18
            term18 = (
                4 * ((1 - Q1) / r1) * ((1 - Q2) / r2) * scrT_r_rp[ir1, iR]
                + 4 * (1 - Q1) * (1 - Q2) * scrL_r_rp[ir1, iR]
            )

            # Term 19
            term19 = (9/4) * (bPhi1 / D1) * (bPhi2 / D2) * f_NL**2 * Omm0**2 * (H0/2997.9)**4 * v_m4_0_0[ir1, iR]

            # Sum all terms
            result[ir1, iR] = prefactor * (
                term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
                term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19
            )

    return result


def compute_Cl_GR_all_ell(I: IntegralCollection, params: ClGRParams,
                          n_workers: Optional[int] = None) -> np.ndarray:
    """
    Compute C_ℓ^GR for all ℓ values in the collection.

    Args:
        I: IntegralCollection with all TwoFAST integrals
        params: ClGRParams with physical parameters
        n_workers: Number of parallel workers (default: number of CPUs)

    Returns:
        np.ndarray: C_ℓ^GR values of size (nr, nR, n_ell)
    """
    nr = len(I.rr)
    nR = len(I.RR)
    nell = len(I.ell_values)
    result = np.zeros((nr, nR, nell))

    if n_workers is None:
        n_workers = os.cpu_count() or 4

    def compute_single_ell(ell_idx_ell):
        ell_idx, ell = ell_idx_ell
        return ell_idx, compute_Cl_GR(I, params, ell)

    # Parallelize over ell values
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for ell_idx, Cl_ell in executor.map(compute_single_ell,
                                             enumerate(I.ell_values)):
            result[:, :, ell_idx] = Cl_ell

    return result


def compute_Cl_GR_terms(I: IntegralCollection, params: ClGRParams, ell: int) -> dict:
    """
    Compute individual terms of C_ℓ^GR for debugging/analysis.

    Returns dict with 2D arrays (nr, nR) for each of the 19 terms.

    Args:
        I: IntegralCollection with pre-computed integrals
        params: ClGRParams with cosmological parameters (arrays on r grid)
        ell: The multipole ℓ value (not index!)

    Returns:
        dict with keys:
        - 'term1' through 'term19': Individual contributions (each is nr × nR array)
        - 'total': Sum of all terms (= C_ℓ^GR)
        - 'density_rsd': Lines 1-4 (standard density and RSD terms)
        - 'doppler_1D': Lines 5-10 (1D integral Doppler terms)
        - 'lensing_2D': Lines 11-12 (2D integral lensing terms)
        - 'fNL_linear': Lines 13-16 (linear f_NL terms)
        - 'fNL_cross': Lines 17-18 (cross terms)
        - 'fNL_squared': Line 19 (f_NL² term)
    """
    ell_idx = _find_ell_idx(I.ell_values, ell)
    nr = len(I.rr)
    nR = len(I.RR)

    # Initialize arrays for each term
    terms = {f'term{i}': np.zeros((nr, nR)) for i in range(1, 20)}

    # Extract scalar parameters
    f_NL = params.f_NL
    Omm0 = params.Omm0
    H0 = params.H0
    fNL_prefactor = 1.5 * Omm0 * (H0 / 2997.9)**2

    # Pre-extract integral slices for this ell
    w_0_0_0 = I[('w', 0, 0, 0, 'none')][:, :, ell_idx]
    w_0_2_0 = I[('w', 0, 2, 0, 'none')][:, :, ell_idx]
    w_0_0_2 = I[('w', 0, 0, 2, 'none')][:, :, ell_idx]
    w_0_2_2 = I[('w', 0, 2, 2, 'none')][:, :, ell_idx]
    w_m1_0_1 = I[('w', -1, 0, 1, 'none')][:, :, ell_idx]
    w_m1_2_1 = I[('w', -1, 2, 1, 'none')][:, :, ell_idx]
    w_m1_1_0 = I[('w', -1, 1, 0, 'none')][:, :, ell_idx]
    w_m1_1_2 = I[('w', -1, 1, 2, 'none')][:, :, ell_idx]
    w_m2_0_0 = I[('w', -2, 0, 0, 'none')][:, :, ell_idx]
    w_m2_2_0 = I[('w', -2, 2, 0, 'none')][:, :, ell_idx]
    w_m2_0_2 = I[('w', -2, 0, 2, 'none')][:, :, ell_idx]
    w_m2_1_1 = I[('w', -2, 1, 1, 'none')][:, :, ell_idx]
    w_m3_1_0 = I[('w', -3, 1, 0, 'none')][:, :, ell_idx]
    w_m3_0_1 = I[('w', -3, 0, 1, 'none')][:, :, ell_idx]
    w_m4_0_0 = I[('w', -4, 0, 0, 'none')][:, :, ell_idx]

    s_m2_0_0_r = I[('s', -2, 0, 0, 'r')][:, :, ell_idx]
    s_m2_0_2_r = I[('s', -2, 0, 2, 'r')][:, :, ell_idx]
    s_m2_0_0_rp = I[('s', -2, 0, 0, 'rp')][:, :, ell_idx]
    s_m2_2_0_rp = I[('s', -2, 2, 0, 'rp')][:, :, ell_idx]
    s_m3_0_1_r = I[('s', -3, 0, 1, 'r')][:, :, ell_idx]
    s_m3_1_0_rp = I[('s', -3, 1, 0, 'rp')][:, :, ell_idx]
    s_m4_0_0_r = I[('s', -4, 0, 0, 'r')][:, :, ell_idx]
    s_m4_0_0_rp = I[('s', -4, 0, 0, 'rp')][:, :, ell_idx]

    t_m2_0_0_r = I[('t', -2, 0, 0, 'r')][:, :, ell_idx]
    t_m2_0_2_r = I[('t', -2, 0, 2, 'r')][:, :, ell_idx]
    t_m2_0_0_rp = I[('t', -2, 0, 0, 'rp')][:, :, ell_idx]
    t_m2_2_0_rp = I[('t', -2, 2, 0, 'rp')][:, :, ell_idx]
    t_m3_0_1_r = I[('t', -3, 0, 1, 'r')][:, :, ell_idx]
    t_m3_1_0_rp = I[('t', -3, 1, 0, 'rp')][:, :, ell_idx]
    t_m4_0_0_r = I[('t', -4, 0, 0, 'r')][:, :, ell_idx]
    t_m4_0_0_rp = I[('t', -4, 0, 0, 'rp')][:, :, ell_idx]

    l_m2_0_0_r = I[('l', -2, 0, 0, 'r')][:, :, ell_idx]
    l_m2_0_2_r = I[('l', -2, 0, 2, 'r')][:, :, ell_idx]
    l_m2_0_0_rp = I[('l', -2, 0, 0, 'rp')][:, :, ell_idx]
    l_m2_2_0_rp = I[('l', -2, 2, 0, 'rp')][:, :, ell_idx]
    l_m3_0_1_r = I[('l', -3, 0, 1, 'r')][:, :, ell_idx]
    l_m3_1_0_rp = I[('l', -3, 1, 0, 'rp')][:, :, ell_idx]
    l_m4_0_0_r = I[('l', -4, 0, 0, 'r')][:, :, ell_idx]
    l_m4_0_0_rp = I[('l', -4, 0, 0, 'rp')][:, :, ell_idx]

    scrX_r_rp = I[('scrX', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrX_rp_r = I[('scrX', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrY_r_rp = I[('scrY', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrY_rp_r = I[('scrY', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrZ_r_rp = I[('scrZ', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrZ_rp_r = I[('scrZ', -4, 0, 0, 'rp_r')][:, :, ell_idx]
    scrS_r_rp = I[('scrS', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrT_r_rp = I[('scrT', -4, 0, 0, 'r_rp')][:, :, ell_idx]
    scrL_r_rp = I[('scrL', -4, 0, 0, 'r_rp')][:, :, ell_idx]

    u_m2_0_0 = I[('u', -2, 0, 0, 'none')][:, :, ell_idx]
    u_m2_2_0 = I[('u', -2, 2, 0, 'none')][:, :, ell_idx]
    u_m2_0_2 = I[('u', -2, 0, 2, 'none')][:, :, ell_idx]
    u_m3_1_0 = I[('u', -3, 1, 0, 'none')][:, :, ell_idx]
    u_m3_0_1 = I[('u', -3, 0, 1, 'none')][:, :, ell_idx]
    u_m4_0_0 = I[('u', -4, 0, 0, 'none')][:, :, ell_idx]

    scrs_m4_r = I[('scrs', -4, 0, 0, 'r')][:, :, ell_idx]
    scrs_m4_rp = I[('scrs', -4, 0, 0, 'rp')][:, :, ell_idx]
    scrt_m4_r = I[('scrt', -4, 0, 0, 'r')][:, :, ell_idx]
    scrt_m4_rp = I[('scrt', -4, 0, 0, 'rp')][:, :, ell_idx]
    scrl_m4_r = I[('scrl', -4, 0, 0, 'r')][:, :, ell_idx]
    scrl_m4_rp = I[('scrl', -4, 0, 0, 'rp')][:, :, ell_idx]

    v_m4_0_0 = I[('v', -4, 0, 0, 'none')][:, :, ell_idx]

    # Loop over all (r, R) grid points
    for iR in range(nR):
        R = I.RR[iR]
        for ir1 in range(nr):
            r1 = I.rr[ir1]
            r2 = R * r1

            # Skip if r2 is out of r grid range
            if r2 < I.rr[0] or r2 > I.rr[-1]:
                continue

            # Get parameters at r1 and r2 (function calls)
            D1, D2 = params.D(r1), params.D(r2)
            H1, H2 = params.H(r1), params.H(r2)
            bg1, bg2 = params.bg(r1), params.bg(r2)
            β1, β2 = params.beta(r1), params.beta(r2)
            B1, B2 = params.B(r1), params.B(r2)
            A1, A2 = params.A(r1), params.A(r2)
            Q1, Q2 = params.Q(r1), params.Q(r2)
            bPhi1, bPhi2 = params.bPhi(r1), params.bPhi(r2)

            prefactor = D1 * D2

            # Term 1
            terms['term1'][ir1, iR] = prefactor * bg1 * bg2 * (
                w_0_0_0[ir1, iR] - β1 * w_0_2_0[ir1, iR]
                - β2 * w_0_0_2[ir1, iR] + β1 * β2 * w_0_2_2[ir1, iR]
            )

            # Term 2
            terms['term2'][ir1, iR] = prefactor * (
                bg1 * H2 * B2 * (w_m1_0_1[ir1, iR] - β1 * w_m1_2_1[ir1, iR])
                + bg2 * H1 * B1 * (w_m1_1_0[ir1, iR] - β2 * w_m1_1_2[ir1, iR])
            )

            # Term 3
            terms['term3'][ir1, iR] = prefactor * bg1 * H2**2 * A2 * (
                w_m2_0_0[ir1, iR] - β1 * w_m2_2_0[ir1, iR]
                + (H1 / bg1) * B1 * w_m3_1_0[ir1, iR]
            )

            # Term 4
            terms['term4'][ir1, iR] = prefactor * bg2 * H1**2 * A1 * (
                w_m2_0_0[ir1, iR] - β2 * w_m2_0_2[ir1, iR]
                + (H2 / bg2) * B2 * w_m3_0_1[ir1, iR]
            )

            # Term 5
            terms['term5'][ir1, iR] = prefactor * bg1 * (B2 / β2) * (
                s_m2_0_0_rp[ir1, iR] - β1 * s_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * s_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * s_m4_0_0_rp[ir1, iR]
            )

            # Term 6
            terms['term6'][ir1, iR] = prefactor * bg2 * (B1 / β1) * (
                s_m2_0_0_r[ir1, iR] - β2 * s_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * s_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * s_m4_0_0_r[ir1, iR]
            )

            # Term 7
            terms['term7'][ir1, iR] = prefactor * (-2) * bg1 * ((1 - Q2) / r2) * (
                t_m2_0_0_rp[ir1, iR] - β1 * t_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * t_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * t_m4_0_0_rp[ir1, iR]
            )

            # Term 8
            terms['term8'][ir1, iR] = prefactor * (-2) * bg2 * ((1 - Q1) / r1) * (
                t_m2_0_0_r[ir1, iR] - β2 * t_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * t_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * t_m4_0_0_r[ir1, iR]
            )

            # Term 9
            terms['term9'][ir1, iR] = prefactor * (-2) * bg1 * (1 - Q2) * (
                l_m2_0_0_rp[ir1, iR] - β1 * l_m2_2_0_rp[ir1, iR]
                + (H1 / bg1) * B1 * l_m3_1_0_rp[ir1, iR]
                + (H1**2 / bg1) * A1 * l_m4_0_0_rp[ir1, iR]
            )

            # Term 10
            terms['term10'][ir1, iR] = prefactor * (-2) * bg2 * (1 - Q1) * (
                l_m2_0_0_r[ir1, iR] - β2 * l_m2_0_2_r[ir1, iR]
                + (H2 / bg2) * B2 * l_m3_0_1_r[ir1, iR]
                + (H2**2 / bg2) * A2 * l_m4_0_0_r[ir1, iR]
            )

            # Term 11
            terms['term11'][ir1, iR] = prefactor * (-2) * bg1 * (1 - Q2) * (
                (B1 / (β1 * r2)) * scrX_r_rp[ir1, iR]
                + (B1 / β1) * scrY_r_rp[ir1, iR]
                - 2 * ((1 - Q1) / (bg1 * r2)) * scrZ_rp_r[ir1, iR]
            )

            # Term 12
            terms['term12'][ir1, iR] = prefactor * (-2) * bg2 * (1 - Q1) * (
                (B2 / (β2 * r1)) * scrX_rp_r[ir1, iR]
                + (B2 / β2) * scrY_rp_r[ir1, iR]
                - 2 * ((1 - Q2) / (bg2 * r1)) * scrZ_r_rp[ir1, iR]
            )

            # Term 13
            terms['term13'][ir1, iR] = prefactor * fNL_prefactor * bg1 * (bPhi2 / D2) * f_NL * (
                u_m2_0_0[ir1, iR] - β1 * u_m2_2_0[ir1, iR]
                + (H1 / bg1) * B1 * u_m3_1_0[ir1, iR]
                + (H1**2 / bg1) * A1 * u_m4_0_0[ir1, iR]
            )

            # Term 14
            terms['term14'][ir1, iR] = prefactor * fNL_prefactor * bg2 * (bPhi1 / D1) * f_NL * (
                u_m2_0_0[ir1, iR] - β2 * u_m2_0_2[ir1, iR]
                + (H2 / bg2) * B2 * u_m3_0_1[ir1, iR]
                + (H2**2 / bg2) * A2 * u_m4_0_0[ir1, iR]
            )

            # Term 15
            terms['term15'][ir1, iR] = prefactor * fNL_prefactor * (bPhi1 / D1) * f_NL * (
                (B2 / (bg2 * β2)) * scrs_m4_rp[ir1, iR]
                - 2 * (1 - Q2) * ((1 / r2) * scrt_m4_rp[ir1, iR] + scrl_m4_rp[ir1, iR])
            )

            # Term 16
            terms['term16'][ir1, iR] = prefactor * fNL_prefactor * (bPhi2 / D2) * f_NL * (
                (B1 / (bg1 * β1)) * scrs_m4_r[ir1, iR]
                - 2 * (1 - Q1) * ((1 / r1) * scrt_m4_r[ir1, iR] + scrl_m4_r[ir1, iR])
            )

            # Term 17
            terms['term17'][ir1, iR] = prefactor * (
                H1 * H2 * B1 * B2 * w_m2_1_1[ir1, iR]
                + H1**2 * H2**2 * A1 * A2 * w_m4_0_0[ir1, iR]
                + (B1 / (bg1 * β1)) * (B2 / (bg2 * β2)) * scrS_r_rp[ir1, iR]
            )

            # Term 18
            terms['term18'][ir1, iR] = prefactor * (
                4 * ((1 - Q1) / r1) * ((1 - Q2) / r2) * scrT_r_rp[ir1, iR]
                + 4 * (1 - Q1) * (1 - Q2) * scrL_r_rp[ir1, iR]
            )

            # Term 19
            terms['term19'][ir1, iR] = prefactor * (9/4) * (bPhi1 / D1) * (bPhi2 / D2) * f_NL**2 * Omm0**2 * (H0/2997.9)**4 * v_m4_0_0[ir1, iR]

    # Compute grouped terms
    terms['total'] = sum(terms[f'term{i}'] for i in range(1, 20))
    terms['density_rsd'] = terms['term1'] + terms['term2'] + terms['term3'] + terms['term4']
    terms['doppler_1D'] = sum(terms[f'term{i}'] for i in range(5, 11))
    terms['lensing_2D'] = terms['term11'] + terms['term12']
    terms['fNL_linear'] = sum(terms[f'term{i}'] for i in range(13, 17))
    terms['fNL_cross'] = terms['term17'] + terms['term18']
    terms['fNL_squared'] = terms['term19']

    return terms


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python calcClGR.py <integrals.h5>")
        sys.exit(1)

    filename = sys.argv[1]
    I = IntegralCollection(filename)

    # Example with array parameters (using from_arrays)
    nr = len(I.rr)
    params = ClGRParams.from_arrays(
        rr=I.rr,
        D=np.ones(nr) * 0.8,
        H=np.ones(nr) * 0.001,
        bg=np.ones(nr) * 1.5,
        beta=np.ones(nr) * 0.4,
        B=np.ones(nr) * 0.1,
        A=np.ones(nr) * 0.05,
        Q=np.ones(nr) * 0.4,
        bPhi=np.ones(nr) * 1.0
    )

    ell = I.ell_values[0]
    Cl = compute_Cl_GR(I, params, ell)
    print(f"\nC_ℓ^GR(ℓ={ell}) shape: {Cl.shape}")
    print(f"C_ℓ^GR range: [{Cl.min():.6e}, {Cl.max():.6e}]")
