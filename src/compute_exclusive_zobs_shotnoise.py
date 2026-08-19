#!/usr/bin/env python3
# =============================================================================
# Build exclusive observed-redshift-bin shot noise for the existing
# continuous SPHEREx z_obs centers.
#
# For each sample s and observed bin i:
#
#   P_i(z_true) =
#       ∫_{zobs_lo_i}^{zobs_hi_i} dz_obs p(z_obs | z_true)
#
#   Nbar_i =
#       ∫ dz_true nbar_s(z_true) P_i(z_true)
#
#   N_ij = delta_ij / Nbar_i
#
# The script also:
#   - verifies that the bins partition the observed-z survey interval;
#   - verifies number conservation;
#   - compares against Nshot_cont.h5 and Nshot_cont_zproj.h5;
#   - compares current point windows with exact bin-integrated windows;
#   - writes CSV summaries and diagnostic plots.
# =============================================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr


CANONICAL_SIGMA_REL = {
    1: 0.003,
    2: 0.010,
    3: 0.030,
    4: 0.100,
    5: 0.200,
}

FLOOR = np.finfo(float).tiny


# =============================================================================
# BASIC HELPERS
# =============================================================================

def integrate(x: np.ndarray, y: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def weighted_cosine(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    numerator = np.sum(weights * x * y)

    denominator = np.sqrt(
        np.sum(weights * x * x)
        * np.sum(weights * y * y)
    )

    return float(
        numerator / max(denominator, FLOOR)
    )


def trap_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if len(x) < 2:
        raise ValueError("Need at least two integration nodes.")

    weights = np.empty_like(x)

    weights[0] = 0.5 * (
        x[1] - x[0]
    )

    weights[-1] = 0.5 * (
        x[-1] - x[-2]
    )

    weights[1:-1] = 0.5 * (
        x[2:] - x[:-2]
    )

    return weights


def decode_string_array(dataset) -> list[str]:
    raw = dataset[()]

    output = []

    for item in np.asarray(raw).ravel():
        if isinstance(item, bytes):
            output.append(
                item.decode(errors="replace")
            )
        else:
            output.append(str(item))

    return output


def resolve_path(
    metadata_file: Path,
    raw_path: str,
) -> Path:
    path = Path(
        raw_path.strip()
    )

    if path.is_absolute():
        return path

    return (
        metadata_file.parent
        / path
    ).resolve()


def safe_eigen_diagnostics(
    matrix: np.ndarray,
) -> dict:
    matrix = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        matrix
    )

    positive = eigenvalues[
        eigenvalues > 0
    ]

    condition = (
        eigenvalues[-1] / positive[0]
        if positive.size
        else np.inf
    )

    return {
        "eigenvalues": eigenvalues,
        "lambda_min": eigenvalues[0],
        "lambda_max": eigenvalues[-1],
        "condition": condition,
        "negative": int(
            np.sum(eigenvalues < 0)
        ),
    }


def correlation_matrix(
    matrix: np.ndarray,
) -> np.ndarray:
    diagonal = np.diag(
        matrix
    )

    scale = np.sqrt(
        np.maximum(
            diagonal,
            0.0,
        )
    )

    denominator = (
        scale[:, None]
        * scale[None, :]
    )

    return np.divide(
        matrix,
        denominator,
        out=np.zeros_like(matrix),
        where=denominator > 0,
    )


# =============================================================================
# PARAMETER-FILE HELPERS
# =============================================================================

def sample_sigma_rel(
    params: h5py.File,
    sample: int,
) -> float:
    sigma_key = f"sigmaz_{sample}"
    zmid_key = f"zmid{sample}"

    if (
        sigma_key in params
        and zmid_key in params
    ):
        sigma_z = np.asarray(
            params[sigma_key],
            dtype=float,
        )

        zmid = np.asarray(
            params[zmid_key],
            dtype=float,
        )

        relative = (
            sigma_z
            / (1.0 + zmid)
        )

        return float(
            np.median(relative)
        )

    return CANONICAL_SIGMA_REL[
        sample
    ]


def load_nbar_true(
    params: h5py.File,
    sample: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return nbar_s(z_true) in the same convention used by the current
    continuous shot-noise generator.

    Preferred:
        nbar(z) = 1 / shot_noise_d{s}(z)

    Fallback:
        use sel_func_{s}_com as the shape, normalized to the total surface
        density 1 / shot_noise_{s}.
    """
    ztest = np.asarray(
        params["ztest"],
        dtype=float,
    )

    dense_key = f"shot_noise_d{sample}"

    if dense_key in params:
        inverse_nbar = np.asarray(
            params[dense_key],
            dtype=float,
        )

        if len(inverse_nbar) != len(ztest):
            raise ValueError(
                f"{dense_key} has length {len(inverse_nbar)}, "
                f"but ztest has length {len(ztest)}."
            )

        nbar = np.zeros_like(
            inverse_nbar
        )

        positive = inverse_nbar > 0

        nbar[positive] = (
            1.0
            / inverse_nbar[positive]
        )

        return (
            ztest,
            nbar,
            dense_key,
        )

    selection_key = (
        f"sel_func_{sample}_com"
    )

    scalar_key = (
        f"shot_noise_{sample}"
    )

    if (
        selection_key in params
        and scalar_key in params
    ):
        selection = np.asarray(
            params[selection_key],
            dtype=float,
        )

        total_surface_density = (
            1.0
            / float(
                np.asarray(
                    params[scalar_key]
                )
            )
        )

        normalization = integrate(
            ztest,
            selection,
        )

        if normalization <= 0:
            raise ValueError(
                f"{selection_key} has nonpositive integral."
            )

        nbar = (
            total_surface_density
            * selection
            / normalization
        )

        return (
            ztest,
            nbar,
            f"{selection_key}+{scalar_key}",
        )

    raise KeyError(
        f"Sample {sample} has neither "
        f"{dense_key} nor "
        f"{selection_key}+{scalar_key}."
    )


def load_selection_shape(
    params: h5py.File,
    sample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    ztest = np.asarray(
        params["ztest"],
        dtype=float,
    )

    key = f"sel_func_{sample}_com"

    if key not in params:
        raise KeyError(
            f"Missing {key}; needed for the point/bin window check."
        )

    selection = np.asarray(
        params[key],
        dtype=float,
    )

    return ztest, selection


# =============================================================================
# OBSERVED-BIN CONSTRUCTION
# =============================================================================

def observed_bin_edges(
    centers: np.ndarray,
    survey_zmin: float,
    survey_zmax: float,
) -> np.ndarray:
    centers = np.asarray(
        centers,
        dtype=float,
    )

    if len(centers) < 2:
        raise ValueError(
            "Need at least two z_obs centers per sample."
        )

    if np.any(
        np.diff(centers) <= 0
    ):
        raise ValueError(
            "z_obs centers must be strictly increasing."
        )

    edges = np.empty(
        len(centers) + 1,
        dtype=float,
    )

    edges[0] = survey_zmin
    edges[-1] = survey_zmax

    edges[1:-1] = 0.5 * (
        centers[:-1]
        + centers[1:]
    )

    if np.any(
        np.diff(edges) <= 0
    ):
        raise ValueError(
            "Constructed observed-z bin edges are not strictly increasing."
        )

    if (
        centers[0] < edges[0]
        or centers[-1] > edges[-1]
    ):
        raise ValueError(
            "Observed-z centers lie outside the requested survey boundaries."
        )

    return edges


def bin_probability(
    z_true: np.ndarray,
    lower: float,
    upper: float,
    sigma_rel: float,
) -> np.ndarray:
    sigma_z = (
        sigma_rel
        * (1.0 + z_true)
    )

    upper_standardized = (
        upper - z_true
    ) / sigma_z

    lower_standardized = (
        lower - z_true
    ) / sigma_z

    probability = (
        ndtr(upper_standardized)
        - ndtr(lower_standardized)
    )

    return np.clip(
        probability,
        0.0,
        1.0,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo",
        default="/storage/group/duj13/default/PowerFull.jl",
    )

    parser.add_argument(
        "--params",
        default=None,
    )

    parser.add_argument(
        "--meta",
        default=None,
    )

    parser.add_argument(
        "--out",
        default=None,
    )

    parser.add_argument(
        "--diag-dir",
        default="/storage/home/gql5196/scratch/diag_exclusive_shotnoise",
    )

    parser.add_argument(
        "--survey-zmin",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--survey-zmax",
        type=float,
        default=4.6,
    )

    parser.add_argument(
        "--nz-true",
        type=int,
        default=32769,
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = parser.parse_args()

    repo = Path(
        args.repo
    )

    params_file = Path(
        args.params
        or repo
        / "spherex_params_opt_gaussian.h5"
    )

    metadata_file = Path(
        args.meta
        or repo
        / "examples"
        / "tracer_meta_cont.h5"
    )

    output_file = Path(
        args.out
        or repo
        / "examples"
        / "Nshot_cont_exclusive.h5"
    )

    diagnostic_directory = Path(
        args.diag_dir
    )

    diagnostic_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        params_file,
        metadata_file,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.nz_true < 4097:
        raise ValueError(
            "--nz-true should be at least 4097."
        )

    print("=" * 110)
    print("EXCLUSIVE OBSERVED-z SHOT-NOISE TEST")
    print("=" * 110)
    print("params :", params_file)
    print("meta   :", metadata_file)
    print("output :", output_file)
    print("diag   :", diagnostic_directory)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    with h5py.File(
        metadata_file,
        "r",
    ) as metadata:
        sample_all = np.asarray(
            metadata["sample"],
            dtype=int,
        )

        zobs_all = np.asarray(
            metadata["z_obs"],
            dtype=float,
        )

        tracer_paths = (
            decode_string_array(
                metadata["tracer_paths"]
            )
            if "tracer_paths" in metadata
            else None
        )

    ntracer = len(
        sample_all
    )

    if len(zobs_all) != ntracer:
        raise ValueError(
            "sample and z_obs lengths differ."
        )

    if (
        tracer_paths is not None
        and len(tracer_paths) != ntracer
    ):
        raise ValueError(
            "tracer_paths length differs from metadata length."
        )

    samples = sorted(
        np.unique(sample_all)
    )

    print(
        "sample counts:",
        {
            int(sample): int(
                np.sum(
                    sample_all == sample
                )
            )
            for sample in samples
        },
    )

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------

    noise = np.zeros(
        (ntracer, ntracer),
        dtype=float,
    )

    zobs_lower = np.empty(
        ntracer,
        dtype=float,
    )

    zobs_upper = np.empty(
        ntracer,
        dtype=float,
    )

    delta_zobs = np.empty(
        ntracer,
        dtype=float,
    )

    nbar_bin = np.empty(
        ntracer,
        dtype=float,
    )

    point_bin_cosine = np.full(
        ntracer,
        np.nan,
    )

    point_bin_l1 = np.full(
        ntracer,
        np.nan,
    )

    point_bin_mean_shift = np.full(
        ntracer,
        np.nan,
    )

    point_bin_width_ratio = np.full(
        ntracer,
        np.nan,
    )

    sample_rows = []
    tracer_rows = []

    parameter_sources: Dict[int, str] = {}
    sigma_rel_values: Dict[int, float] = {}

    # -------------------------------------------------------------------------
    # Sample loop
    # -------------------------------------------------------------------------

    with h5py.File(
        params_file,
        "r",
    ) as params:
        ztest_file = np.asarray(
            params["ztest"],
            dtype=float,
        )

        true_zmin = max(
            args.survey_zmin,
            float(ztest_file[0]),
        )

        true_zmax = min(
            args.survey_zmax,
            float(ztest_file[-1]),
        )

        z_true = np.linspace(
            true_zmin,
            true_zmax,
            args.nz_true,
        )

        true_weights = trap_weights(
            z_true
        )

        for sample in samples:
            sample = int(
                sample
            )

            global_indices = np.where(
                sample_all == sample
            )[0]

            order = np.argsort(
                zobs_all[global_indices]
            )

            sorted_global_indices = (
                global_indices[order]
            )

            centers = zobs_all[
                sorted_global_indices
            ]

            edges = observed_bin_edges(
                centers,
                args.survey_zmin,
                args.survey_zmax,
            )

            sigma_rel = sample_sigma_rel(
                params,
                sample,
            )

            sigma_rel_values[
                sample
            ] = sigma_rel

            z_nbar_table, nbar_table, source = load_nbar_true(
                params,
                sample,
            )

            parameter_sources[
                sample
            ] = source

            nbar_true = np.interp(
                z_true,
                z_nbar_table,
                nbar_table,
                left=0.0,
                right=0.0,
            )

            z_selection_table, selection_table = load_selection_shape(
                params,
                sample,
            )

            selection_true = np.interp(
                z_true,
                z_selection_table,
                selection_table,
                left=0.0,
                right=0.0,
            )

            sigma_z = (
                sigma_rel
                * (1.0 + z_true)
            )

            survey_capture_probability = (
                ndtr(
                    (
                        args.survey_zmax
                        - z_true
                    )
                    / sigma_z
                )
                - ndtr(
                    (
                        args.survey_zmin
                        - z_true
                    )
                    / sigma_z
                )
            )

            total_true_surface_density = np.sum(
                true_weights
                * nbar_true
            )

            expected_observed_surface_density = np.sum(
                true_weights
                * nbar_true
                * survey_capture_probability
            )

            bin_probability_sum = np.zeros_like(
                z_true
            )

            sample_bin_counts = []

            for local_index, global_index in enumerate(
                sorted_global_indices
            ):
                lower = float(
                    edges[local_index]
                )

                upper = float(
                    edges[local_index + 1]
                )

                probability = bin_probability(
                    z_true,
                    lower,
                    upper,
                    sigma_rel,
                )

                bin_probability_sum += probability

                count = np.sum(
                    true_weights
                    * nbar_true
                    * probability
                )

                if not np.isfinite(count) or count <= 0:
                    raise ValueError(
                        f"S{sample}, tracer {global_index + 1}: "
                        f"nonpositive bin count {count}."
                    )

                sample_bin_counts.append(
                    count
                )

                zobs_lower[
                    global_index
                ] = lower

                zobs_upper[
                    global_index
                ] = upper

                delta_zobs[
                    global_index
                ] = (
                    upper - lower
                )

                nbar_bin[
                    global_index
                ] = count

                noise[
                    global_index,
                    global_index,
                ] = 1.0 / count

                # -------------------------------------------------------------
                # Compare point-likelihood window to exact bin-integrated window
                # -------------------------------------------------------------

                exact_bin_raw = (
                    selection_true
                    * probability
                )

                exact_bin_norm = np.sum(
                    true_weights
                    * exact_bin_raw
                )

                if exact_bin_norm <= 0:
                    raise ValueError(
                        f"S{sample}, tracer {global_index + 1}: "
                        "exact bin window has nonpositive normalization."
                    )

                exact_bin_phi = (
                    exact_bin_raw
                    / exact_bin_norm
                )

                if tracer_paths is not None:
                    tracer_path = resolve_path(
                        metadata_file,
                        tracer_paths[
                            global_index
                        ],
                    )

                    if not tracer_path.is_file():
                        raise FileNotFoundError(
                            tracer_path
                        )

                    with h5py.File(
                        tracer_path,
                        "r",
                    ) as tracer:
                        tracer_z = np.asarray(
                            tracer["z"],
                            dtype=float,
                        )

                        tracer_phi = np.asarray(
                            tracer["phi"],
                            dtype=float,
                        )

                    point_phi = np.interp(
                        z_true,
                        tracer_z,
                        tracer_phi,
                        left=0.0,
                        right=0.0,
                    )

                    point_norm = np.sum(
                        true_weights
                        * point_phi
                    )

                    if point_norm <= 0:
                        raise ValueError(
                            f"Tracer {global_index + 1} has "
                            "nonpositive point-window normalization."
                        )

                    point_phi /= point_norm

                    cosine = weighted_cosine(
                        point_phi,
                        exact_bin_phi,
                        true_weights,
                    )

                    l1_difference = np.sum(
                        true_weights
                        * np.abs(
                            point_phi
                            - exact_bin_phi
                        )
                    )

                    point_mean = np.sum(
                        true_weights
                        * z_true
                        * point_phi
                    )

                    bin_mean = np.sum(
                        true_weights
                        * z_true
                        * exact_bin_phi
                    )

                    point_variance = np.sum(
                        true_weights
                        * (
                            z_true
                            - point_mean
                        ) ** 2
                        * point_phi
                    )

                    bin_variance = np.sum(
                        true_weights
                        * (
                            z_true
                            - bin_mean
                        ) ** 2
                        * exact_bin_phi
                    )

                    point_bin_cosine[
                        global_index
                    ] = cosine

                    point_bin_l1[
                        global_index
                    ] = l1_difference

                    point_bin_mean_shift[
                        global_index
                    ] = (
                        bin_mean - point_mean
                    )

                    point_bin_width_ratio[
                        global_index
                    ] = np.sqrt(
                        max(bin_variance, 0.0)
                        / max(
                            point_variance,
                            FLOOR,
                        )
                    )

                tracer_rows.append({
                    "tracer_id": global_index + 1,
                    "sample": sample,
                    "z_obs": zobs_all[
                        global_index
                    ],
                    "zobs_lo": lower,
                    "zobs_hi": upper,
                    "delta_zobs": upper - lower,
                    "nbar_bin_sr": count,
                    "Nii_exclusive": 1.0 / count,
                    "point_bin_cosine": point_bin_cosine[
                        global_index
                    ],
                    "point_bin_L1": point_bin_l1[
                        global_index
                    ],
                    "point_bin_mean_shift": point_bin_mean_shift[
                        global_index
                    ],
                    "point_bin_width_ratio": point_bin_width_ratio[
                        global_index
                    ],
                })

            sample_bin_counts = np.asarray(
                sample_bin_counts
            )

            partition_error = np.max(
                np.abs(
                    bin_probability_sum
                    - survey_capture_probability
                )
            )

            summed_bin_surface_density = np.sum(
                sample_bin_counts
            )

            count_closure = (
                summed_bin_surface_density
                / expected_observed_surface_density
                - 1.0
            )

            capture_fraction = (
                expected_observed_surface_density
                / total_true_surface_density
                if total_true_surface_density > 0
                else np.nan
            )

            sample_cosines = point_bin_cosine[
                sorted_global_indices
            ]

            sample_l1 = point_bin_l1[
                sorted_global_indices
            ]

            sample_rows.append({
                "sample": sample,
                "Nbin": len(
                    sorted_global_indices
                ),
                "sigma_rel": sigma_rel,
                "nbar_source": source,
                "total_true_surface_density_sr": total_true_surface_density,
                "observed_surface_density_sr": expected_observed_surface_density,
                "sum_bin_surface_density_sr": summed_bin_surface_density,
                "capture_fraction": capture_fraction,
                "count_closure": count_closure,
                "max_partition_error": partition_error,
                "min_delta_zobs": np.min(
                    np.diff(edges)
                ),
                "median_delta_zobs": np.median(
                    np.diff(edges)
                ),
                "max_delta_zobs": np.max(
                    np.diff(edges)
                ),
                "min_Nii": np.min(
                    1.0 / sample_bin_counts
                ),
                "median_Nii": np.median(
                    1.0 / sample_bin_counts
                ),
                "max_Nii": np.max(
                    1.0 / sample_bin_counts
                ),
                "median_point_bin_cosine": np.nanmedian(
                    sample_cosines
                ),
                "min_point_bin_cosine": np.nanmin(
                    sample_cosines
                ),
                "median_point_bin_L1": np.nanmedian(
                    sample_l1
                ),
                "max_point_bin_L1": np.nanmax(
                    sample_l1
                ),
            })

            print(
                f"S{sample}: Nbin={len(sorted_global_indices):3d}, "
                f"sigma_rel={sigma_rel:.6f}, "
                f"capture={capture_fraction:.8f}, "
                f"closure={count_closure:+.3e}, "
                f"partition={partition_error:.3e}, "
                f"median Nii={np.median(1.0/sample_bin_counts):.6e}, "
                f"median window cosine={np.nanmedian(sample_cosines):.8f}"
            )

    # -------------------------------------------------------------------------
    # Write output HDF5
    # -------------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        output_file,
        "w",
    ) as output:
        output["N"] = noise
        output["sample"] = sample_all
        output["z_obs"] = zobs_all

        output["zobs_lo"] = zobs_lower
        output["zobs_hi"] = zobs_upper
        output["delta_zobs"] = delta_zobs

        output["nbar_bin"] = nbar_bin
        output["nbar_eff"] = nbar_bin

        output["fsky"] = 1.0

        window_group = output.create_group(
            "window_check"
        )

        window_group["point_bin_cosine"] = point_bin_cosine
        window_group["point_bin_L1"] = point_bin_l1
        window_group["point_bin_mean_shift"] = point_bin_mean_shift
        window_group["point_bin_width_ratio"] = point_bin_width_ratio

        provenance = output.create_group(
            "provenance"
        )

        provenance["noise_type"] = (
            "exclusive observed-z bins; diagonal Poisson"
        )

        provenance["formula"] = (
            "Nii=1/int dz_true nbar_s(z_true) "
            "int_bin_i dz_obs p(z_obs|z_true)"
        )

        provenance["params"] = str(
            params_file
        )

        provenance["meta"] = str(
            metadata_file
        )

        provenance["survey_zmin"] = args.survey_zmin
        provenance["survey_zmax"] = args.survey_zmax
        provenance["nz_true"] = args.nz_true

        provenance["bin_edges"] = (
            "midpoints between existing z_obs centers; "
            "outer edges fixed to survey bounds"
        )

        provenance["offdiagonal_noise"] = 0
        provenance["includes_fsky_factor"] = 0
        provenance["signal_compatibility"] = (
            "diagnostic midpoint approximation; current Cl uses point "
            "p(z_obs_center|z_true), not exact bin-integrated windows"
        )

        for sample in samples:
            provenance[
                f"sigma_rel_{sample}"
            ] = sigma_rel_values[
                int(sample)
            ]

            provenance[
                f"nbar_source_{sample}"
            ] = parameter_sources[
                int(sample)
            ]

    print("\nWrote:")
    print(output_file)

    # -------------------------------------------------------------------------
    # CSV outputs
    # -------------------------------------------------------------------------

    sample_dataframe = pd.DataFrame(
        sample_rows
    )

    tracer_dataframe = pd.DataFrame(
        tracer_rows
    ).sort_values(
        [
            "sample",
            "z_obs",
        ]
    )

    sample_csv = (
        diagnostic_directory
        / "exclusive_shotnoise_sample_summary.csv"
    )

    tracer_csv = (
        diagnostic_directory
        / "exclusive_shotnoise_tracer_summary.csv"
    )

    sample_dataframe.to_csv(
        sample_csv,
        index=False,
    )

    tracer_dataframe.to_csv(
        tracer_csv,
        index=False,
    )

    print("\nSample summary:")
    print(
        sample_dataframe.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Compare with old noise matrices
    # -------------------------------------------------------------------------

    comparison_rows = []

    comparison_files = {
        "old": (
            repo
            / "examples"
            / "Nshot_cont.h5"
        ),
        "zproj": (
            repo
            / "examples"
            / "Nshot_cont_zproj.h5"
        ),
        "exclusive": output_file,
    }

    loaded_matrices = {}

    for label, path in comparison_files.items():
        if not path.is_file():
            print(
                f"Skipping comparison file {label}: {path}"
            )
            continue

        with h5py.File(
            path,
            "r",
        ) as handle:
            matrix = np.asarray(
                handle["N"],
                dtype=float,
            )

            file_sample = np.asarray(
                handle["sample"],
                dtype=int,
            )

            file_zobs = np.asarray(
                handle["z_obs"],
                dtype=float,
            )

        if not np.array_equal(
            file_sample,
            sample_all,
        ):
            raise ValueError(
                f"{label}: sample ordering differs."
            )

        if not np.allclose(
            file_zobs,
            zobs_all,
            rtol=0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"{label}: z_obs ordering differs."
            )

        loaded_matrices[
            label
        ] = 0.5 * (
            matrix + matrix.T
        )

    exclusive_diagonal = np.diag(
        noise
    )

    for sample in samples:
        sample = int(
            sample
        )

        indices = np.where(
            sample_all == sample
        )[0]

        exclusive_block = noise[
            np.ix_(
                indices,
                indices,
            )
        ]

        exclusive_eigen = safe_eigen_diagnostics(
            exclusive_block
        )

        for label, matrix in loaded_matrices.items():
            block = matrix[
                np.ix_(
                    indices,
                    indices,
                )
            ]

            eigen = safe_eigen_diagnostics(
                block
            )

            ratio = (
                exclusive_diagonal[
                    indices
                ]
                / np.maximum(
                    np.diag(block),
                    FLOOR,
                )
            )

            offdiagonal = (
                block
                - np.diag(
                    np.diag(block)
                )
            )

            comparison_rows.append({
                "sample": sample,
                "reference": label,
                "Ntracer": len(
                    indices
                ),
                "median exclusive/reference diag": np.median(
                    ratio
                ),
                "p05 exclusive/reference diag": np.percentile(
                    ratio,
                    5,
                ),
                "p95 exclusive/reference diag": np.percentile(
                    ratio,
                    95,
                ),
                "reference offdiag/Frobenius": (
                    np.linalg.norm(
                        offdiagonal
                    )
                    / max(
                        np.linalg.norm(block),
                        FLOOR,
                    )
                ),
                "reference condition": eigen[
                    "condition"
                ],
                "exclusive condition": exclusive_eigen[
                    "condition"
                ],
                "reference negative eigenvalues": eigen[
                    "negative"
                ],
                "exclusive negative eigenvalues": exclusive_eigen[
                    "negative"
                ],
            })

    comparison_dataframe = pd.DataFrame(
        comparison_rows
    )

    comparison_csv = (
        diagnostic_directory
        / "exclusive_vs_existing_noise.csv"
    )

    comparison_dataframe.to_csv(
        comparison_csv,
        index=False,
    )

    print("\nNoise comparison:")
    print(
        comparison_dataframe.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(14, 16),
        dpi=160,
    )

    for row, sample in enumerate(
        samples
    ):
        sample = int(
            sample
        )

        indices = np.where(
            sample_all == sample
        )[0]

        order = np.argsort(
            zobs_all[indices]
        )

        indices = indices[
            order
        ]

        z = zobs_all[
            indices
        ]

        for label, matrix in loaded_matrices.items():
            axes[row, 0].semilogy(
                z,
                np.diag(matrix)[
                    indices
                ],
                label=label,
            )

        axes[row, 0].set_ylabel(
            f"S{sample}\n$N_{{ii}}$"
        )

        axes[row, 0].grid(
            alpha=0.25,
            which="both",
        )

        axes[row, 0].legend(
            fontsize=8,
        )

        if "zproj" in loaded_matrices:
            ratio = (
                exclusive_diagonal[
                    indices
                ]
                / np.maximum(
                    np.diag(
                        loaded_matrices[
                            "zproj"
                        ]
                    )[
                        indices
                    ],
                    FLOOR,
                )
            )

            axes[row, 1].semilogy(
                z,
                ratio,
            )

            axes[row, 1].axhline(
                1.0,
                linestyle="--",
                linewidth=1,
            )

        axes[row, 1].set_ylabel(
            f"S{sample}\n"
            r"$N_{ii}^{\rm exclusive}/N_{ii}^{\rm zproj}$"
        )

        axes[row, 1].grid(
            alpha=0.25,
            which="both",
        )

    axes[-1, 0].set_xlabel(
        r"$z_{\rm obs}$"
    )

    axes[-1, 1].set_xlabel(
        r"$z_{\rm obs}$"
    )

    axes[0, 0].set_title(
        "Diagonal noise comparison"
    )

    axes[0, 1].set_title(
        "Exclusive / zproj diagonal ratio"
    )

    plt.tight_layout()

    noise_plot = (
        diagnostic_directory
        / "exclusive_noise_comparison.png"
    )

    plt.savefig(
        noise_plot,
        bbox_inches="tight",
    )

    if args.show:
        plt.show()
    else:
        plt.close()

    fig, axes = plt.subplots(
        len(samples),
        2,
        figsize=(14, 16),
        dpi=160,
    )

    for row, sample in enumerate(
        samples
    ):
        sample = int(
            sample
        )

        indices = np.where(
            sample_all == sample
        )[0]

        order = np.argsort(
            zobs_all[indices]
        )

        indices = indices[
            order
        ]

        z = zobs_all[
            indices
        ]

        axes[row, 0].plot(
            z,
            point_bin_cosine[
                indices
            ],
        )

        axes[row, 0].axhline(
            0.99,
            linestyle="--",
            linewidth=1,
        )

        axes[row, 0].set_ylim(
            0.8,
            1.002,
        )

        axes[row, 0].set_ylabel(
            f"S{sample}\ncosine"
        )

        axes[row, 0].grid(
            alpha=0.25,
        )

        axes[row, 1].plot(
            z,
            point_bin_l1[
                indices
            ],
        )

        axes[row, 1].set_ylabel(
            f"S{sample}\nL1 difference"
        )

        axes[row, 1].grid(
            alpha=0.25,
        )

    axes[-1, 0].set_xlabel(
        r"$z_{\rm obs}$"
    )

    axes[-1, 1].set_xlabel(
        r"$z_{\rm obs}$"
    )

    axes[0, 0].set_title(
        "Point-window versus finite-bin window cosine"
    )

    axes[0, 1].set_title(
        r"$\int dz\,|\phi_{\rm point}-\phi_{\rm bin}|$"
    )

    plt.tight_layout()

    window_plot = (
        diagnostic_directory
        / "point_vs_bin_windows.png"
    )

    plt.savefig(
        window_plot,
        bbox_inches="tight",
    )

    if args.show:
        plt.show()
    else:
        plt.close()

    print("\nWrote diagnostics:")
    print(" ", sample_csv)
    print(" ", tracer_csv)
    print(" ", comparison_csv)
    print(" ", noise_plot)
    print(" ", window_plot)

    # -------------------------------------------------------------------------
    # Automated verdict
    # -------------------------------------------------------------------------

    print("\n" + "=" * 110)
    print("AUTOMATED CHECKS")
    print("=" * 110)

    maximum_closure = np.max(
        np.abs(
            sample_dataframe[
                "count_closure"
            ]
        )
    )

    maximum_partition = np.max(
        sample_dataframe[
            "max_partition_error"
        ]
    )

    minimum_cosine = np.nanmin(
        point_bin_cosine
    )

    median_cosine = np.nanmedian(
        point_bin_cosine
    )

    print(
        f"max count closure error   = {maximum_closure:.3e}"
    )

    print(
        f"max probability partition = {maximum_partition:.3e}"
    )

    print(
        f"median point/bin cosine   = {median_cosine:.8f}"
    )

    print(
        f"minimum point/bin cosine  = {minimum_cosine:.8f}"
    )

    if maximum_closure > 1.0e-6:
        print(
            "WARNING: observed-bin counts do not close accurately."
        )

    if maximum_partition > 1.0e-10:
        print(
            "WARNING: observed bins do not partition the survey interval."
        )

    if minimum_cosine >= 0.99:
        print(
            "Point windows are extremely close to the finite-bin windows; "
            "using the existing Cl matrices with this noise is a strong "
            "midpoint approximation."
        )
    elif median_cosine >= 0.98:
        print(
            "Point and finite-bin windows are broadly similar, but the "
            "exclusive-noise Fisher remains an approximate diagnostic."
        )
    else:
        print(
            "WARNING: point and finite-bin windows differ materially. "
            "The exact production test requires rebuilding tracer phi(z) "
            "with bin-integrated photo-z probabilities and rerunning Step 3."
        )


if __name__ == "__main__":
    main()
