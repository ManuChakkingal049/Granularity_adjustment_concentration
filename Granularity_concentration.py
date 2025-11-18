import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta

# ------------------------------------------------------------------------------------
# ------------------------------ Helper Functions ------------------------------------
# ------------------------------------------------------------------------------------

def get_rho(PD, asset_class):
    """Basel II correlation function."""
    asset_class = asset_class.str.lower()
    return np.where(
        asset_class == "corporate",
        0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) +
        0.24 * np.exp(-50 * PD) / (1 - np.exp(-50)),
        np.where(
            asset_class == "retail", 0.03,
            np.where(asset_class == "sovereign", 0.12, 0.12)
        )
    )

def parse_date(x):
    """Flexible date parser that handles DD/MM/YYYY and YYYY-MM-DD."""
    try:
        return pd.to_datetime(x, dayfirst=True)
    except:
        return pd.to_datetime(x, dayfirst=False)


def maturity_adjustment(PD, maturity_years):
    """Basel II maturity adjustment."""
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    return (1 + (maturity_years - 2.5) * b) / (1 - 1.5 * b)


def vasicek_capital(df, run_date, confidence_level):
    """Calculate Vasicek unexpected loss capital."""
    PD = df['PD']
    LGD = df['LGD']
    EAD = df['EAD']
    asset_class = df['AssetClass']

    maturity_years = ((df['MaturityDate'] - run_date).dt.days / 365).clip(lower=0.01)

    rho = get_rho(PD, asset_class)
    MA = maturity_adjustment(PD, maturity_years)

    z_alpha = norm.ppf(confidence_level)
    z_PD = norm.ppf(PD)

    vasicek_cap_req = LGD * (
        norm.cdf((z_PD + np.sqrt(rho) * z_alpha) / np.sqrt(1 - rho)) - PD
    ) * MA

    expected_loss = PD * LGD * EAD
    unexpected_loss = vasicek_cap_req * EAD
    default_loss_rate = PD * LGD

    return expected_loss, unexpected_loss, vasicek_cap_req, default_loss_rate, MA


def granularity_adjustment(df, vasicek_cap_req, default_loss_rate, delta, gamma):
    """Calculate complete granularity adjustment."""
    total_EAD = df['EAD'].sum()
    exposure_fraction = df['EAD'] / total_EAD

    lgd_variance = gamma * df['LGD'] * (1 - df['LGD'])
    sensitivity_factor = (lgd_variance + df['LGD'] ** 2) / df['LGD']

    K_star_total = (exposure_fraction * vasicek_cap_req).sum()

    term1 = delta * sensitivity_factor * (vasicek_cap_req + default_loss_rate)
    term2 = delta * (vasicek_cap_req + default_loss_rate) ** 2 * (lgd_variance / df['LGD'] ** 2)
    term3 = vasicek_cap_req * (
        sensitivity_factor + 2 * (vasicek_cap_req + default_loss_rate) *
        (lgd_variance / df['LGD'] ** 2)
    )

    ga_per_obligor = (1 / (2 * K_star_total)) * (exposure_fraction ** 2) * (term1 + term2 - term3)
    total_GA = ga_per_obligor.sum()

    return total_GA, ga_per_obligor, lgd_variance, sensitivity_factor, exposure_fraction, K_star_total


# ------------------------------------------------------------------------------------
# ------------------------------ Streamlit UI ----------------------------------------
# ------------------------------------------------------------------------------------

st.title("Basel II/III Granularity Adjustment (GA) Calculator")

st.write("""
This tool calculates:

### ✔ Expected Loss (EL)  
### ✔ Unexpected Loss (UL) under the Vasicek IRB Model  
### ✔ Granularity Adjustment (GA) including LGD variance  
### ✔ GA Capital Charge & GA RWA  

---

### 🔍 **Granularity Adjustment — Simple Explanation**

Banks face higher concentration risk when:

- **One exposure is too large** (high *exposure fraction s*)  
- **LGD is volatile** (high *LGD variance*)  
- **The sensitivity factor (C)** rises due to increased LGD uncertainty  
- **Defaults become more correlated**, increasing *(K + R)*  

The granularity formula increases capital for these concentration effects.

---
""")

# ------------------------------------------------------------
# ---------- Default Dataset (example for template) ----------
# ------------------------------------------------------------

default_data = pd.DataFrame({
    "CustomerID": [101, 102, 103, 104, 105],
    "PD": [0.02, 0.01, 0.015, 0.025, 0.018],
    "LGD": [0.45, 0.40, 0.35, 0.50, 0.55],
    "EAD": [80_000_000, 120_000_000, 90_000_000, 100_000_000, 50_000_000],
    "AssetClass": ["Corporate"] * 5,
    "MaturityDate": [datetime.today() + timedelta(days=365*2.5)] * 5
})

# ------------------------------------------------------------
# ----------------------- CSV Upload -------------------------
# ------------------------------------------------------------

st.subheader("📁 Upload CSV or Use Default Data")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df["MaturityDate"] = df["MaturityDate"].apply(parse_date)
else:
    st.info("Using built-in dataset as example.")
    df = default_data.copy()

# ------------------------------------------------------------
# ----------------------- User Inputs -------------------------
# ------------------------------------------------------------

run_date = st.date_input("Run Date", datetime.today())
confidence_level = st.number_input("Confidence Level", value=0.999, format="%.3f")
gamma = st.number_input("LGD Variance Gamma", value=0.25, format="%.4f")
delta = st.number_input("Delta (Concentration Sensitivity)", value=4.83, format="%.2f")

capital_ratio = st.number_input(
    "Capital Ratio (Denominator for RWA, typically 0.08)",
    value=0.08, format="%.4f"
)

# ------------------------------------------------------------
# ----------------------- Calculations ------------------------
# ------------------------------------------------------------

EL, UL, K, R, MA = vasicek_capital(df, pd.to_datetime(run_date), confidence_level)

df["Expected_Loss"] = EL
df["Unexpected_Loss"] = UL
df["Vasicek_Capital"] = K
df["Default_Loss_Rate"] = R
df["Maturity_Adjustment"] = MA

total_GA, GA_per_obligor, lgd_variance, sensitivity_factor, exposure_fraction, K_star_total = \
    granularity_adjustment(df, K, R, delta, gamma)

df["LGD_Variance"] = lgd_variance
df["Sensitivity_Factor_C"] = sensitivity_factor
df["Exposure_Fraction"] = exposure_fraction
df["K_Star_Total"] = K_star_total
df["Granularity_Adjustment"] = GA_per_obligor

# Portfolio-level GA Capital Charge → GA RWA
total_EAD = df["EAD"].sum()
GA_capital_charge = total_GA * total_EAD
GA_RWA = GA_capital_charge / capital_ratio

# ------------------------------------------------------------
# ----------------------- Display Output ----------------------
# ------------------------------------------------------------

st.subheader("📊 Detailed Per-Obligor Results")
st.dataframe(df)

st.subheader("📈 Portfolio Summary")
st.write(f"**Total Granularity Adjustment (GA):** {total_GA:.6f}")
st.write(f"**GA Capital Charge:** {GA_capital_charge:,.2f}")
st.write(f"**GA RWA:** {GA_RWA:,.2f}")

# ------------------------------------------------------------
# ----------------------- Excel Export -----------------------
# ------------------------------------------------------------

st.subheader("📥 Export Results to Excel")

filename = st.text_input("Excel filename (without .xlsx)", value="GA_Output")

if st.button("Download Excel"):
    output_file = f"{filename}.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Detailed_Results", index=False)

        pd.DataFrame({
            "Total_GA": [total_GA],
            "GA_Capital_Charge": [GA_capital_charge],
            "GA_RWA": [GA_RWA]
        }).to_excel(writer, sheet_name="Portfolio_Summary", index=False)

    with open(output_file, "rb") as f:
        st.download_button(
            "Download Excel File",
            data=f,
            file_name=output_file
        )

st.success("Calculation completed successfully!")

