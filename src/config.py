"""Frozen economic, delay, and benchmark constants from experimental_protocol.md."""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EconomicConfig:
    dispute_fee: float = 15.00       # c_cb ($15.00 per auto-approved fraud)
    review_cost: float = 2.50        # c_r ($2.50 per manual review or Stage-0 audit)
    friction_cost: float = 0.50      # c_f ($0.50 per false-positive review/audit)
    recovery_rate: float = 0.90      # q (90% fraud value recovered when reviewed)
    cap_percentile: float = 99.0     # A_cap percentile (top 1% routed to mandatory review)
    num_strata: int = 5              # K = 5 quantile amount strata


@dataclass(frozen=True)
class ControllerConfig:
    stage_0_lag: int = 1             # \Delta_0 = 1 day (fast investigator audit turnaround)
    maturity_floor: float = 0.50     # G_min = 0.50 minimum IPCW maturity floor
    c_gain_base: float = 0.50        # Arora et al. dead-time gain constant (eta_base = 0.5 / \Delta_1)
    eta_dual: float = 0.15           # Two-speed critically damped gain (\eta \in [0.10, 0.25])
    q_audit: float = 0.01            # Default 1.0% Stage-0 diagnostic audit budget
    e_min: float = 0.005             # Positivity floor (0.5%)
    dr_beta: float = 1.0             # Doubly-robust control-variate weight (\beta = 1.0)
    reanchor_period: int = 30        # Periodic re-anchoring interval R = 30 days
    theta_max_mult: float = 10.0     # Anti-windup cap: \theta_max = log(10 * \alpha)


@dataclass(frozen=True)
class BenchmarkSplitConfig:
    name: str
    mean_amount_ref: float
    cap_amount_ref: float
    uncontrolled_loss_per_1k: float
    budget_fraction: float
    delta_grid: List[int] = field(default_factory=list)
    g_hat_table: List[float] = field(default_factory=list)
    shock_day: int = 300

    @property
    def target_loss_per_1k(self) -> float:
        return self.uncontrolled_loss_per_1k * self.budget_fraction

    @property
    def alpha_normalized(self) -> float:
        return (self.target_loss_per_1k / 1000.0) / self.mean_amount_ref


SPARKOV_CONFIG = BenchmarkSplitConfig(
    name="sparkov_24m",
    mean_amount_ref=70.93,
    cap_amount_ref=540.00,
    uncontrolled_loss_per_1k=2577.23,
    budget_fraction=0.05,  # $128.86 per 1k txns
    delta_grid=[15, 30, 45, 60, 90],
    g_hat_table=[0.52, 0.74, 0.88, 0.95, 1.00],
    shock_day=300,
)

ULB_CONFIG = BenchmarkSplitConfig(
    name="ulb_stream",
    mean_amount_ref=88.35,
    cap_amount_ref=1098.00,
    uncontrolled_loss_per_1k=201.12,
    budget_fraction=0.20,  # $40.22 per 1k txns
    delta_grid=[7, 14, 21, 30, 60],
    g_hat_table=[0.52, 0.75, 0.89, 0.96, 1.00],
    shock_day=110,
)

ECONOMIC_CFG = EconomicConfig()
CONTROLLER_CFG = ControllerConfig()
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]