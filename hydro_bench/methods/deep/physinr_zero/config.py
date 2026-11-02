"""Every hyperparameter of PhysINR-Zero in one auditable place.

The methodology fixes some quantities (Fourier frequencies ``L = 4``, feature
channels ``16/24/32``, ``tau_max = 8``, ``kappa_A = 0.05``) and leaves the rest
unspecified.  Where the methodology is silent, this file records the value we
chose, the reason, and the section of the methodology the choice implements.
Every field is also written into the provenance record of each run, so a
manuscript claim is never separated from the configuration that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass(frozen=True)
class PhysINRConfig:
    """Immutable configuration.  Mutate by ``dataclasses.replace``."""

    # ---------------------------------------------------------- architecture
    # Methodology §3: three depthwise-separable blocks, 16/24/32 channels,
    # spatial resolution reduced by no more than 4x.
    feature_channels: Tuple[int, int, int] = (16, 24, 32)
    feature_strides: Tuple[int, int, int] = (2, 2, 1)
    leaky_slope: float = 0.1

    # Methodology §4: compact two-layer MLP on [f(x), gamma(x)].
    mlp_hidden: int = 64
    fourier_levels: int = 4          # L
    # Total trainable parameters must stay below this budget (§4, §22).
    param_budget: int = 25_000

    # ------------------------------------------------------ physical constants
    # Methodology §5.  tau_max = 8 is fixed by the methodology; delta_max bounds
    # how far the spatial field may depart from the Jerlov prior (value chosen
    # here, see Assumption A1 in docs/METHODOLOGY_RESOLUTION.md).
    tau_max: float = 8.0
    delta_max: float = 1.5
    depth_max: float = 15.0          # d_max, bound on the effective depth scale
    # Methodology §7.  kappa_A = 0.05 is fixed by the methodology.
    ambient_min: float = 0.10
    ambient_max: float = 0.90
    kappa_ambient: float = 0.05
    # Methodology §8.  alpha_max and kappa_alpha bound the nonlinear curve
    # (values chosen here, see Assumption A2).
    alpha_max: float = 2.0
    kappa_alpha: float = 0.25
    # Methodology §9.  Small residual on the re-degradation operator.
    residual_max: float = 0.05
    eps: float = 1e-4                # transmission denominator guard

    # Number of Jerlov water types in the soft assignment (§6).  All ten
    # standard types are used: oceanic I, IA, IB, II, III and coastal 1,3,5,7,9.
    num_water_types: int = 10

    # ------------------------------------------------------------- test-time
    # Methodology §19: adaptation steps between 5 and 20; the feature extractor
    # learns at 0.1x the field learning rate.
    adapt_steps: int = 12
    lr_field: float = 3e-3
    lr_extractor: float = 3e-4       # = 0.1 * lr_field (§19)
    # Methodology §20: warm-up length for the cosine ramp of the secondary
    # losses, expressed in adaptation steps.
    warmup_steps: int = 5
    # Methodology §4: high Fourier frequencies are switched on progressively.
    fourier_ramp: int = 4            # steps to fully activate one octave

    # ---------------------------------------------------------------- losses
    # Group weights of L_total (§10).  phys and cycle are held at their maximum
    # from step 0; the other four follow the cosine schedule of §20.
    w_phys: float = 1.0
    w_color: float = 0.5
    w_struct: float = 0.5
    w_cycle: float = 0.5
    w_spatial: float = 0.05
    w_jerlov: float = 0.3
    w_prior: float = 0.2

    # Sub-weights (§11-§17).  Each is documented in the methodology section it
    # implements.
    w_range: float = 1.0             # §11
    w_grad: float = 0.1              # §12
    w_std: float = 0.5               # §13
    w_ssim: float = 1.0              # §14.1
    w_edge: float = 0.3              # §14.2
    w_entropy: float = 0.1           # §14.3
    w_noise: float = 0.1             # §14.3
    w_curv: float = 0.05             # §15
    w_alpha_tv: float = 0.1          # §15
    w_dev: float = 0.1               # §16
    w_water: float = 0.01            # §16, increased over adaptation
    w_ratio: float = 0.1             # §17.2
    w_curve: float = 0.1             # §17.3
    w_magnitude: float = 0.01        # §17.3

    # §11 Charbonnier.
    charbonnier_eps: float = 1e-3

    # §14.2 reliability mask uses the 95th percentile of the input gradient.
    edge_quantile: float = 0.95
    # §14.3 local-entropy estimator: Gaussian-bin soft histogram.
    entropy_window: int = 7
    entropy_bins: int = 16
    entropy_bin_sigma: float = 0.08
    # §14.3 noise penalty is evaluated on locally flat regions only.
    flat_window: int = 7
    # §15 edge-aware total variation.
    tv_edge_kappa: float = 10.0
    # §17.1 identity constraint; delta_min is deliberately small.
    identity_delta: float = 0.02

    # ---- the colour-cast refactor ----------------------------------------
    # Two new loss groups, each ablatable as a unit through ``drop_group`` and
    # each internally weighted so a single term can be ablated on its own --
    # the same group/sub-weight split the seven original groups use.
    #
    # "cast" = w_cast * (w_lab * L_lab + w_cc * L_cc).
    # L_lab is the CIELAB cast loss.  "symmetric" is |mu_a*| + |mu_b*|: it pulls
    # every cast, red included, toward neutral.  "signed" penalises only the
    # green/blueward half of each axis (a* < 0 is green, b* < 0 is blue), which
    # is the failure mode this refactor targets and leaves warm casts alone.
    # L_cc is the gray-world colour-constancy term, a weak channel-balance
    # prior on the restored image.
    w_cast: float = 1.0
    w_lab: float = 0.3
    lab_mode: str = "signed"             # "symmetric" | "signed"
    w_cc: float = 0.2
    # "bound" = w_bound * (w_bound_order * L_bound + w_ambient_prior *
    # L_ambient).  L_bound constrains the channel ordering of the optical
    # thickness.  With ``bound_margin_from_table`` the allowed ordering is the
    # one the *assigned water type* exhibits in the Jerlov table; the universal
    # clear-water ordering beta_R > beta_G >= beta_B is an option, but the
    # table violates it for oceanic III and every coastal type (see
    # physics._ORDERING_CAVEAT), so it is not the default.  L_ambient pulls the
    # learned ambient field toward the data-driven background-light estimate.
    w_bound: float = 1.0
    w_bound_order: float = 0.1
    bound_margin_from_table: bool = True
    w_ambient_prior: float = 0.05

    # ---- spectral parameterisation (§C2) ---------------------------------
    # Integrate the Jerlov table against a Gaussian channel response instead
    # of reading three wavelengths.  ``False`` reproduces the single-wavelength
    # interpolation of Assumption A3 exactly.
    use_spectral_beta: bool = True
    spectral_sigma_nm: float = 30.0

    # ---- adaptive background light (§B2) ---------------------------------
    # The ambient field's per-channel range is bracketed around an estimate
    # from the image instead of the fixed global [ambient_min, ambient_max].
    # ``ambient_adaptive = False`` restores the fixed bracket (legacy mode).
    ambient_adaptive: bool = True
    ambient_dark_fraction: float = 0.10  # red-deepest pixels = deep region
    ambient_percentile: float = 0.90     # per-channel value taken there
    # Half-width of the bracket around the data-driven estimate.  Calibrated
    # on UIEB visual-check frames (experiments/visual_check/sweep_ambient.py):
    # the shipped 0.25 let the cycle loss drive the learned ambient to the
    # degenerate J = 0 solution on deep-coastal frames (red deleted, cast
    # survives); 0.10 keeps the ambient at the estimator's measurement and
    # recovers up to +4 dB PSNR on the worst frame without hurting the rest.
    ambient_window: float = 0.10
    # Lower clamp of the ADAPTIVE bracket.  The 0.10 legacy floor was chosen
    # for the fixed bracket; clamping the data-driven bracket at the same
    # value fights the estimator exactly on the deep-coastal frames the
    # method targets: UIEB frames routinely estimate a red background light
    # of 0.02-0.08 (measured on the visual-check frames), and with the floor
    # at 0.10 the inverse (I - A(1-t))/t goes negative on the red channel
    # wherever t is at its 0.25 floor -- the red channel is deleted, and the
    # cyan/green cast survives the correction.  0.05 keeps the ambient
    # physically non-negative even at the full kappa_ambient tanh offset
    # (0.05 - 0.05 = 0) while letting the estimate speak.
    ambient_floor_adaptive: float = 0.05

    # ---- YIQ chroma alignment (§D) ---------------------------------------
    # A parameter-free post-process: the I/Q amplitudes are scaled by gains
    # derived from the output's own chroma statistics, so luminance is
    # untouched and colourful scenes are spared.  It carries no parameters, so
    # the 25k budget and the existing meta-initialisation both survive.
    yiq_align: bool = True
    yiq_strength: float = 0.5            # how hard a detected bias is shrunk
    yiq_chroma_floor: float = 0.6        # gains never fall below this
    yiq_sigma_ref: float = 0.05          # chroma spread below which = bias
    # Which frame the gains are measured on.  "input" reads the degraded frame,
    # so the alignment attacks the cast the water put in; "restored" reads the
    # model's own output, so it also cancels whatever cast the restoration left
    # behind -- the case on clear water, where the coastal-weighted prior adds
    # a cast that was never in the input.  "restored" is the default because
    # the A/B on UIEB/UCCS found it strictly better on clear water and a
    # no-op where the input already carries the cast (experiments/gains_source_ab.py).
    yiq_gains_source: str = "restored"

    # ---- display normalisation (§E) ----------------------------------------
    # A reference-free post-process applied at RENDER time only: each channel's
    # [p_low, p_high] percentile range of the rendered frame is mapped to
    # [0, 1].  The physical inverse is conservative -- the transmission floor
    # caps the amplification, so the output keeps a compressed histogram and
    # the water tint sits mid-scale instead of spanning it -- and supervised
    # competitors effectively learn this display transform.  Calibrated on the
    # UIEB visual-check frames (experiments/visual_check/prototype_display.py):
    # +1.6 dB mean PSNR and frame-100 contrast 0.19 -> 0.28, the reference's
    # own level.  It touches the rendered output only -- never the adaptation
    # loss and never the re-degradation cycle -- so the reference-free
    # objective, the physics, and the meta-initialisation are all unchanged;
    # ``display_norm = False`` reproduces the pre-§E rendering exactly.
    display_norm: bool = True
    display_p_low: float = 0.01
    display_p_high: float = 0.99
    display_strength: float = 1.0

    # ------------------------------------------------- water-type head (§6 fix)
    # The published head is Linear(1, K) fed by the spatial mean of the field's
    # single water-logit channel.  With no content path and no supervision its
    # optimum is a constant -- the marginal type distribution of the training
    # tasks -- which is the collapse the diagnostics measured (rotation moves
    # the weights by ~3e-4; the type is effectively a checkpoint constant).
    # "physics" replaces the bottleneck with a content descriptor: the pooled
    # field logit, the channel means, their log-ratios, the estimated
    # per-channel background light, and the pooled extractor features.
    type_head_features: str = "physics"
    # Meta-training knows the type each synthetic task was drawn with, so the
    # outer loop can supervise the head directly.  Two signals, weighted by
    # how identifiable each is from a single frame: cross-entropy against the
    # drawn type mixture (a weak anchor -- type and depth are confounded, so
    # even a perfect head tops out well below 1.0), and log-space MSE on the
    # optical thickness the prior implies, tau_jerlov, against the task's
    # tau = beta * depth (identifiable to R^2 ~ 0.4 from global features --
    # the level of murkiness is what a single frame actually shows).  Without
    # any of this the head's optimum is the marginal type distribution -- a
    # constant, which is the collapse the diagnostics measured.  The inner
    # adaptation loop stays unsupervised -- the zero-shot claim depends on
    # that -- so at test time the head reads only the image.
    w_type_sup: float = 0.1
    w_tau_sup: float = 0.5
    # ---- v2: the prior's level is data-driven (root-cause fix) -----------
    # v1's pathology: the meta-initialised tau prior over-estimates the
    # optical thickness on real images (domain gap -- it is fitted to
    # synthetic degradations), and the §8 inverse divides by t = exp(-tau),
    # so a mid-tone channel with a prior of tau = 1.1 is amplified 3x and
    # 70% of its pixels saturate -- the neon green/cyan wash.  Two changes:
    # the transmission used for the division is floored at ``t_min``
    # (amplification capped at 1/t_min, here 4x), and the prior's *level*
    # comes from the image itself -- range(I_c) = t_c * range(J_c), so the
    # per-channel percentile range gives t_c directly -- with the Jerlov
    # table reduced to a learned spectral-ratio correction on top.
    t_min: float = 0.25
    tau_data_weight: float = 1.0
    # The learned Jerlov component contributes its spectral shape about the
    # channel mean, not a DC level: its DC is the synthetic-tied offset the
    # supervision trained it on (the reference channel's attenuation, invisible
    # to the range estimator), and carrying that DC onto real frames
    # over-amplifies clear water.
    tau_jerlov_centered: bool = True
    # ---- v3: the level head ------------------------------------------------
    # v2's remaining defect: the range estimator normalises the best channel to
    # t = 1, but on real frames even the least attenuated channel sits at
    # t ~ 0.59 (measured against UIEB reference pairs).  The estimator gets the
    # spectral RATIO right (mean error 0.05) and the LEVEL wrong by a factor
    # ~1.7, so the prior under-corrects and the output stays washed out and
    # desaturated.  The level must not be a tuned constant (that would leak the
    # test set); it is a per-image scalar the model learns to predict from the
    # same content descriptor the type head reads, supervised in the outer loop
    # on the synthetic task's own t_best = max_c exp(-beta_c * d).
    # tau_level_weight scales that supervision; 0 disables v3 and reverts to v2.
    level_head: bool = True
    w_level_sup: float = 0.5

    # --------------------------------------------------------------- meta-train
    # Methodology §18.  FOMAML (first-order MAML) is used for the outer loop;
    # the inner loop runs the *unsupervised* L_total, the outer loss uses the
    # synthetic ground truth (see Assumption A5 in the methodology resolution).
    meta_inner_steps: int = 4
    meta_inner_lr: float = 1e-2
    meta_outer_lr: float = 1e-3
    meta_batch: int = 4
    meta_tasks: int = 4_000
    meta_crop: int = 192
    # Synthetic task sampling ranges (§18 factor list).
    # The depth range is calibrated to the optical thickness real underwater
    # photographs carry.  At the former (0.5, 12) m the drawn t_best =
    # max_c exp(-beta_c d) averaged 0.32 with a p10 of 0.01 -- far murkier
    # than real frames, whose best channel sits at t ~ 0.59 (measured by
    # fitting I = J t + A(1-t) on UIEB input/reference pairs).  A prior
    # trained on that distribution systematically over-restores.  Capping at
    # 4 m puts the synthetic mean at 0.58 against the same table, matching the
    # real level from the physics rather than from the evaluation set.
    synth_depth_range: Tuple[float, float] = (0.5, 4.0)
    synth_beta_jitter: float = 0.15       # lognormal sigma on per-channel beta
    synth_ambient_range: Tuple[float, float] = (0.15, 0.85)
    synth_noise_sigma: Tuple[float, float] = (0.0, 0.04)
    synth_blur_sigma: Tuple[float, float] = (0.0, 1.2)
    synth_local_tau: float = 0.35         # amplitude of spatial tau variation
    synth_color_shift: float = 0.10       # per-channel gain/offset jitter
    # Coastal water types (1,3,5,7,9) are where the green/cyan cast lives: heavy
    # backscattering and CDOM absorption from phytoplankton and organic matter.
    # The sampler weights them up by this factor so the meta-initialisation
    # sees the degradation family it must correct, rather than the clear oceanic
    # types that dominate a uniform draw over the ten.
    synth_coastal_weight: float = 0.6
    # CDOM (gelbstoff) absorption at 440 nm, m^-1.  Drawn per task and added to
    # the attenuation: a_cdom(lambda) = a440 exp(-S(lambda-440)) removes blue
    # first, which is the spectral signature of the cast.
    synth_cdom_a440: Tuple[float, float] = (0.0, 0.40)
    synth_cdom_slope: float = 0.015       # S, nm^-1
    # Tie the ambient (asymptotic backscatter) colour to the drawn water type.
    # Physically the backscatter colour IS the water's own transmitted colour
    # -- the reason deep coastal water is green and deep oceanic water blue --
    # so a uniform per-channel draw makes the type unidentifiable from the
    # image and the type head's optimum collapses to the marginal regardless
    # of architecture.  With the tie, A_inf ~ brightness * exp(-beta d_ref).
    synth_ambient_tied: bool = True
    synth_ambient_depth_ref: float = 2.0  # m, reference depth for the colour
    synth_ambient_jitter: float = 0.10    # lognormal sigma on the colour

    # ------------------------------------------------------------ reproducib.
    seed: int = 1234
    device: str = "auto"                  # auto -> mps if available else cpu

    # --------------------------------------------------------------- runtime
    # Coordinate stride used while adapting (methodology §21 step 4a: "sample
    # normalised image coordinates x").  The field is evaluated at every
    # stride-th pixel during the adaptation steps; the enhanced image is
    # always rendered on the full grid.  Because ``_grid`` assigns strided
    # pixels the same coordinates they would have at stride 1, this evaluates
    # the *same* continuous field on a subset of the pixels -- an
    # approximation in sampling, not a different model.  1 = no subsampling;
    # see the stride ablation (docs/METHODOLOGY_RESOLUTION.md, A7).
    loss_stride: int = 2

    # Maximum side length at adaptation time.  Images are *not* resized for the
    # final output (geometry must be preserved for the full-reference metrics);
    # this only guards against pathological inputs during meta-training.
    max_side: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def legacy(cls, **overrides) -> "PhysINRConfig":
        """The pre-refactoring configuration, for reproducibility and ablation.

        Disables every intervention of the colour-cast refactor: the spectral
        parameterisation, the adaptive background-light bracket, the YIQ
        alignment, and the three new loss groups.  What remains is the
        objective and the physics the published numbers were produced with, so
        a before/after comparison isolates the change rather than mixing it
        with an unrelated regression.
        """
        base = {
            "use_spectral_beta": False,
            "ambient_adaptive": False,
            "yiq_align": False,
            "display_norm": False,
            "w_cast": 0.0,
            "w_bound": 0.0,
            "type_head_features": "none",
            "w_type_sup": 0.0,
            "w_tau_sup": 0.0,
            "t_min": 0.0,
            "tau_data_weight": 0.0,
            "tau_jerlov_centered": False,
            "level_head": False,
            "w_level_sup": 0.0,
            # And the synthesis interventions: the published checkpoint was
            # meta-trained on the uniform task family, so reproducing it
            # requires the sampler that produced it -- including the deep
            # depth range that the v4 calibration corrected.
            "synth_coastal_weight": 0.0,
            "synth_cdom_a440": (0.0, 0.0),
            "synth_ambient_tied": False,
            "synth_depth_range": (0.5, 12.0),
        }
        base.update(overrides)
        return cls(**base)

    def parameter_groups(self) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Names of the field/physical parameters vs the feature extractor.

        Used for the two learning-rate levels of §19.
        """
        field_params = (
            "mlp.", "jerlov_head.", "depth_logit", "depth_head.", "level_head.",
        )
        extractor_params = ("feature_extractor.",)
        return field_params, extractor_params


DEFAULT_CONFIG = PhysINRConfig()
