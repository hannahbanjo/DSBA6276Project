from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent

AIRROI_CANDIDATES = [
    REPO_DIR / "airroi.csv",
    REPO_DIR.parent / "airroi.csv",
]

CALENDAR_CANDIDATES = [
    REPO_DIR / "calendar_rates.csv",
    REPO_DIR.parent / "calendar_rates.csv",
]

BUSINESS_LABELS = {
    0: "Stable Core",
    1: "Premium Upside",
}

OUTPUT_TIER_PROFILES = PROJECT_DIR / "cluster_tier_profiles_week2.csv"
OUTPUT_DEMAND_SUMMARY = PROJECT_DIR / "cluster_demand_period_summary_week2.csv"
OUTPUT_SIMULATION = PROJECT_DIR / "portfolio_mix_simulation_week3.csv"
OUTPUT_RECOMMENDATIONS = PROJECT_DIR / "portfolio_recommendations_week3.csv"
OUTPUT_REPORT = PROJECT_DIR / "laasya_week2_week3_report.md"


def resolve_path(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{searched}")


def load_listing_frame() -> pd.DataFrame:
    airroi_path = resolve_path(AIRROI_CANDIDATES, "airroi.csv")
    airroi = pd.read_csv(airroi_path).drop_duplicates(subset="listing_id").copy()
    clusters = pd.read_csv(PROJECT_DIR / "listing_clusters_k2.csv")

    listing = clusters.merge(
        airroi[
            [
                "listing_id",
                "host_id",
                "ttm_revenue",
                "ttm_revpar",
                "l90d_revenue",
                "l90d_avg_rate",
                "l90d_occupancy",
            ]
        ],
        on="listing_id",
        how="left",
    )

    listing["cluster_label"] = listing["cluster_id"].map(BUSINESS_LABELS)
    return listing


def load_calendar_frame() -> pd.DataFrame:
    calendar_path = resolve_path(CALENDAR_CANDIDATES, "calendar_rates.csv")
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    clusters = pd.read_csv(PROJECT_DIR / "listing_clusters_k2.csv")[["listing_id", "cluster_id"]]
    calendar = calendar.merge(clusters, on="listing_id", how="inner")
    calendar["cluster_label"] = calendar["cluster_id"].map(BUSINESS_LABELS)
    calendar["year_month"] = calendar["date"].dt.to_period("M").astype(str)
    return calendar


def build_tier_profiles(listing: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    listing_monthly = (
        calendar.groupby(["cluster_id", "cluster_label", "listing_id", "year_month"], as_index=False)
        .agg(
            monthly_revenue=("revenue", "mean"),
            monthly_occupancy=("occupancy", "mean"),
            monthly_adr=("rate_avg", "mean"),
        )
    )

    listing_risk = (
        listing_monthly.groupby(["cluster_id", "cluster_label", "listing_id"], as_index=False)
        .agg(
            avg_monthly_revenue=("monthly_revenue", "mean"),
            revenue_std_monthly=("monthly_revenue", "std"),
        )
    )
    listing_risk["revenue_cv"] = (
        listing_risk["revenue_std_monthly"] / listing_risk["avg_monthly_revenue"]
    )
    listing_risk["risk_adjusted_return"] = (
        listing_risk["avg_monthly_revenue"] / listing_risk["revenue_std_monthly"]
    )
    listing_risk = listing_risk.fillna(0)

    cluster_profile = (
        listing.groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            listings=("listing_id", "count"),
            guests_mean=("guests", "mean"),
            bedrooms_mean=("bedrooms", "mean"),
            baths_mean=("baths", "mean"),
            cleaning_fee_mean=("cleaning_fee", "mean"),
            reviews_mean=("num_reviews_x", "mean"),
            rating_mean=("rating_overall", "mean"),
            superhost_share=("superhost", "mean"),
            professional_mgmt_share=("professional_management", "mean"),
            adr_mean=("ttm_avg_rate", "mean"),
            occupancy_mean=("ttm_occupancy", "mean"),
            revpar_mean=("ttm_revpar", "mean"),
            ttm_revenue_mean=("ttm_revenue", "mean"),
            l90d_revenue_mean=("l90d_revenue", "mean"),
        )
    )

    risk_profile = (
        listing_risk.groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            avg_monthly_revenue=("avg_monthly_revenue", "mean"),
            revenue_std_monthly=("revenue_std_monthly", "mean"),
            revenue_cv=("revenue_cv", "mean"),
            risk_adjusted_return=("risk_adjusted_return", "mean"),
        )
    )

    demand_profile = (
        calendar.groupby(["cluster_id", "cluster_label", "demand_period"], as_index=False)
        .agg(
            demand_period_adr=("rate_avg", "mean"),
            demand_period_occupancy=("occupancy", "mean"),
            demand_period_revenue=("revenue", "mean"),
        )
    )

    shoulder = (
        demand_profile[demand_profile["demand_period"] == "shoulder"][
            ["cluster_id", "demand_period_revenue"]
        ]
        .rename(columns={"demand_period_revenue": "shoulder_revenue"})
    )
    off_peak = (
        demand_profile[demand_profile["demand_period"] == "off-peak"][
            ["cluster_id", "demand_period_revenue"]
        ]
        .rename(columns={"demand_period_revenue": "off_peak_revenue"})
    )

    profile = cluster_profile.merge(risk_profile, on=["cluster_id", "cluster_label"], how="left")
    profile = profile.merge(shoulder, on="cluster_id", how="left")
    profile = profile.merge(off_peak, on="cluster_id", how="left")
    profile["shoulder_to_off_peak_revenue_lift"] = (
        profile["shoulder_revenue"] - profile["off_peak_revenue"]
    )
    profile["share_of_inventory"] = profile["listings"] / profile["listings"].sum()

    ordered_columns = [
        "cluster_id",
        "cluster_label",
        "listings",
        "share_of_inventory",
        "guests_mean",
        "bedrooms_mean",
        "baths_mean",
        "cleaning_fee_mean",
        "reviews_mean",
        "rating_mean",
        "superhost_share",
        "professional_mgmt_share",
        "adr_mean",
        "occupancy_mean",
        "revpar_mean",
        "ttm_revenue_mean",
        "l90d_revenue_mean",
        "avg_monthly_revenue",
        "revenue_std_monthly",
        "revenue_cv",
        "risk_adjusted_return",
        "off_peak_revenue",
        "shoulder_revenue",
        "shoulder_to_off_peak_revenue_lift",
    ]
    profile = profile[ordered_columns].round(4)

    demand_profile = demand_profile.sort_values(["cluster_id", "demand_period"]).reset_index(drop=True)
    demand_profile = demand_profile.round(4)

    return profile, demand_profile


def build_portfolio_simulation(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stable = profile.loc[profile["cluster_label"] == "Stable Core"].iloc[0]
    premium = profile.loc[profile["cluster_label"] == "Premium Upside"].iloc[0]

    rows = []
    for total_units in range(10, 21):
        for premium_units in range(total_units + 1):
            stable_units = total_units - premium_units

            expected_ttm_revenue = (
                stable_units * stable["ttm_revenue_mean"] + premium_units * premium["ttm_revenue_mean"]
            )
            expected_annual_volatility = math.sqrt(
                stable_units * (stable["revenue_std_monthly"] ** 2)
                + premium_units * (premium["revenue_std_monthly"] ** 2)
            ) * math.sqrt(12)
            weighted_adr = (
                stable_units * stable["adr_mean"] + premium_units * premium["adr_mean"]
            ) / total_units
            weighted_occupancy = (
                stable_units * stable["occupancy_mean"] + premium_units * premium["occupancy_mean"]
            ) / total_units
            weighted_revpar = (
                stable_units * stable["revpar_mean"] + premium_units * premium["revpar_mean"]
            ) / total_units
            expected_off_peak_revenue = (
                stable_units * stable["off_peak_revenue"] + premium_units * premium["off_peak_revenue"]
            ) * 12
            risk_adjusted_score = (
                expected_ttm_revenue / expected_annual_volatility if expected_annual_volatility else 0
            )

            rows.append(
                {
                    "total_units": total_units,
                    "stable_core_units": stable_units,
                    "premium_upside_units": premium_units,
                    "premium_share": premium_units / total_units,
                    "expected_ttm_revenue": expected_ttm_revenue,
                    "expected_annual_volatility": expected_annual_volatility,
                    "risk_adjusted_score": risk_adjusted_score,
                    "portfolio_adr": weighted_adr,
                    "portfolio_occupancy": weighted_occupancy,
                    "portfolio_revpar": weighted_revpar,
                    "expected_off_peak_revenue": expected_off_peak_revenue,
                }
            )

    simulation = pd.DataFrame(rows).round(4)

    recommendations = simulation[
        (
            ((simulation["total_units"] == 10) & (simulation["premium_upside_units"] == 3))
            | ((simulation["total_units"] == 15) & (simulation["premium_upside_units"] == 5))
            | ((simulation["total_units"] == 20) & (simulation["premium_upside_units"] == 6))
        )
    ].copy()
    recommendations["recommendation_note"] = [
        "70/30 mix keeps the portfolio occupancy-led while adding premium upside.",
        "A one-third premium sleeve lifts revenue without making volatility dominant.",
        "At 20 units, six premium homes adds upside while fourteen stable homes anchor resilience.",
    ]

    return simulation, recommendations


def write_report(profile: pd.DataFrame, demand_summary: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    stable = profile.loc[profile["cluster_label"] == "Stable Core"].iloc[0]
    premium = profile.loc[profile["cluster_label"] == "Premium Upside"].iloc[0]

    demand_pivot = demand_summary.pivot(
        index="cluster_label",
        columns="demand_period",
        values="demand_period_revenue",
    )

    report_lines = [
        "# Laasya Week 2 and Week 3 Deliverables",
        "",
        "## Week 2: Cluster Tier Profiles",
        "",
        "Two business-facing tiers were built from the locked k=2 clustering output:",
        "",
        f"- **Stable Core (Cluster 0):** {int(stable['listings'])} listings ({stable['share_of_inventory']:.1%} of inventory), "
        f"ADR ${stable['adr_mean']:.0f}, occupancy {stable['occupancy_mean']:.1%}, RevPAR ${stable['revpar_mean']:.0f}. "
        f"This tier is smaller-format inventory with stronger occupancy, lower revenue volatility, and a higher superhost share.",
        f"- **Premium Upside (Cluster 1):** {int(premium['listings'])} listings ({premium['share_of_inventory']:.1%} of inventory), "
        f"ADR ${premium['adr_mean']:.0f}, occupancy {premium['occupancy_mean']:.1%}, RevPAR ${premium['revpar_mean']:.0f}. "
        f"This tier is larger-format inventory with higher revenue potential, but also higher monthly volatility.",
        "",
        "## Demand-Period Takeaways",
        "",
        f"- Stable Core average monthly revenue rises from ${demand_pivot.loc['Stable Core', 'off-peak']:.0f} in off-peak periods "
        f"to ${demand_pivot.loc['Stable Core', 'shoulder']:.0f} in shoulder periods.",
        f"- Premium Upside average monthly revenue rises from ${demand_pivot.loc['Premium Upside', 'off-peak']:.0f} in off-peak periods "
        f"to ${demand_pivot.loc['Premium Upside', 'shoulder']:.0f} in shoulder periods.",
        "- Both tiers monetize best in shoulder periods, which lines up with the team's temporal demand findings.",
        "",
        "## Week 3: Portfolio Recommendation",
        "",
        "For a 10 to 20 unit host, the most balanced recommendation is to keep roughly 30% of units in Premium Upside "
        "and 70% in Stable Core. That mix sacrifices some top-end revenue versus an all-premium portfolio, but it preserves "
        "occupancy stability, lowers annualized revenue volatility, and keeps off-peak revenue more resilient.",
        "",
        "Recommended mixes:",
        "",
    ]

    for _, row in recommendations.iterrows():
        report_lines.append(
            f"- **{int(row['total_units'])} units:** {int(row['stable_core_units'])} Stable Core + "
            f"{int(row['premium_upside_units'])} Premium Upside | expected TTM revenue ${row['expected_ttm_revenue']:,.0f}, "
            f"annualized volatility ${row['expected_annual_volatility']:,.0f}, risk-adjusted score {row['risk_adjusted_score']:.2f}."
        )

    report_lines.extend(
        [
            "",
            "## Risk-Adjusted Rationale",
            "",
            f"- Stable Core has the better average risk-adjusted return ({stable['risk_adjusted_return']:.2f} vs {premium['risk_adjusted_return']:.2f}).",
            f"- Premium Upside has the stronger revenue ceiling (${premium['ttm_revenue_mean']:,.0f} vs ${stable['ttm_revenue_mean']:,.0f} average TTM revenue per listing), "
            "but its monthly revenue is materially more variable.",
            "- A blended portfolio gives the host a reliable occupancy base while still participating in higher-rate premium demand.",
        ]
    )

    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    listing = load_listing_frame()
    calendar = load_calendar_frame()
    tier_profiles, demand_summary = build_tier_profiles(listing, calendar)
    simulation, recommendations = build_portfolio_simulation(tier_profiles)

    tier_profiles.to_csv(OUTPUT_TIER_PROFILES, index=False)
    demand_summary.to_csv(OUTPUT_DEMAND_SUMMARY, index=False)
    simulation.to_csv(OUTPUT_SIMULATION, index=False)
    recommendations.to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    write_report(tier_profiles, demand_summary, recommendations)

    print(f"Wrote {OUTPUT_TIER_PROFILES}")
    print(f"Wrote {OUTPUT_DEMAND_SUMMARY}")
    print(f"Wrote {OUTPUT_SIMULATION}")
    print(f"Wrote {OUTPUT_RECOMMENDATIONS}")
    print(f"Wrote {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()
