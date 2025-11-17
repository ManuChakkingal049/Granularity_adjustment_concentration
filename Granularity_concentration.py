import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import io

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def get_rho(PD, asset_class):
    """Basel Corporate Correlation Function"""
    return np.where(
        asset_class.str.lower() == 'corporate',
        0.12 * (1 - np.exp(-50 * PD)) / (1 - np.exp(-50)) +
        0.24 * np.exp(-50 * PD) / (1 - np.exp(-50)),
        0.12
    )

def maturity_adjustment(PD, M):
    """Basel maturity adjustment formula."""
    b = (0.11852 - 0.05478 * np.log(PD)) ** 2
    return (1 + (M - 2.5) * b) / (1 - 1.5 * b)

def vasicek_capital(df, run_date, confidence):
    """Compute Vasicek capital and all components."""
    M_years = ((df["MaturityDate"] - run_date).dt.days / 365).clip(lower=0.01)
    PD = df["PD"]
    LGD = df["LGD"]
    EAD = df["EAD"]
    asset_class = df["AssetClass"]

    R = get_rho(PD, asset_class)
    MA = maturity_adjustment(PD, M_years)

    N_inv = norm.ppf(confidence)
    PD_inv = norm.ppf(PD)

    K = LGD * (
        norm.cdf((PD_inv + np.sqrt(R) * N_inv) / np.sqrt(1 - R)) - PD
    ) * MA

    EL = PD * LGD * EAD
    UL = K * EAD
    R_val = PD * LGD

    return EL, UL, K, R_val, MA, M_years, R

def granularity_adjustment(df, K, R, delta, gamma):
    """Full GA formula including LGD variance and exposure share effects."""
    total_EAD = df["EAD"].sum()
    s = df["EAD"] / total_EAD

    Var_LGD = gamma * df["LGD"] * (1 - df["LGD"])
    C = (Var_LGD + df["LGD"] ** 2) / df["LGD"]

    K_star_total = (s * K).sum()

    term1 = delta * C * (K + R)
    term2 = delta * (K + R) ** 2 * Var_LGD / (df["LGD"] ** 2)
    term3 = K * (C + 2 * (K + R) * Var_LGD / (df["LGD"] ** 2))

    GA = (1 / (2 * K_star_total)) * s ** 2 * (term1 + term2 - term3)
    total_GA = GA.sum()

    return total_GA, GA, Var_LGD, C, s, K_star_total, total_EAD

def calc_GA_capital_RWA(total_GA, total_EAD, charge_pct):
    """Compute GA capital charge and corresponding RWA."""
    GA_capital = total_GA * total_EAD
    GA_RWA = GA_capital / charge_pct
    return GA_capital, GA_RWA

# -----------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------

st.title("📘 Credit Risk Capital Calculator with Granularity Adjustment")

st.markdown("""
## 🔍 Methodology Summary

### **Vasicek Risk Capital**
The Vasicek model estimates unexpected loss capital requirement based on:
- **PD**: Probability of Default  
- **LGD**: Loss Given Default  
- **EAD**: Exposure at Default  
- **R**: Asset Correlation  
- **MA**: Maturity Adjustment  

---

## **Granularity Adjustment (GA) – Interpretation in Simple Terms**

Real credit portfolios are not perfectly diversified.  
Large exposures or high-risk loans cause **concentration**, increasing portfolio risk.

GA increases capital if:

### ✔ 1. Exposure Share is High (sᵢ = EADᵢ / ΣEAD)
Large borrowers → more concentration → higher GA.

### ✔ 2. LGD Variability is High
Uncertainty in LGD increases risk:
\[
Var(LGD) = \gamma \cdot LGD(1-LGD)
\]

Higher γ → higher GA.

### ✔ 3. High Sensitivity to Systematic Risk (K + R)
If systemic default risk is high, GA becomes larger.

---

Upload your dataset or use the **default example**.
""")

# -----------------------------------------------------------
# Upload or default dataset
# -----------------------------------------------------------

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("Using built-in example dataset.")
    df = pd.DataFrame({
        "CustomerID": [5001, 5002, 5003, 5004, 5005],
        "PD": [0.02, 0.01, 0.03, 0.015, 0.005],
        "LGD": [0.45, 0.50, 0.40, 0.35, 0.55],
        "EAD": [150e6, 200e6, 120e6, 180e6, 90e6],
        "AssetClass": ["Corporate"] * 5
    })
    df["MaturityDate"] = pd.to_datetime(datetime.today() + timedelta(days=365*2.5))

if "MaturityDate" in df.columns:
    df["MaturityDate"] = pd.to_datetime(df["MaturityDate"])
else:
    st.error("CSV must contain column: MaturityDate")
    st.stop()

# -----------------------------------------------------------
# Inputs
# -----------------------------------------------------------

st.subheader("📌 Model Parameters")

run_date = st.date_input("Run Date", datetime.today())
confidence = st.number_input("Confidence Level", value=0.999)
gamma = st.number_input("LGD Variance Factor (γ)", value=0.25)
delta = st.number_input("Delta Factor (δ)", value=4.83)
charge_pct = st.number_input("Charge to EAD Percentage", value=0.08)

# -----------------------------------------------------------
# Calculations
# -----------------------------------------------------------

EL, UL, K, R_val, MA, M_years, R_asset_corr = vasicek_capital(
    df, pd.to_datetime(run_date), confidence
)

df["MaturityYears"] = M_years
df["AssetCorrelation_R"] = R_asset_corr
df["MaturityAdjustment_MA"] = MA
df["ExpectedLoss_EL"] = EL
df["UnexpectedLoss_UL"] = UL
df["CapitalRequirement_K"] = K
df["LGD_Sensitivity_R"] = R_val

total_GA, GA, Var_LGD, C, s, K_star_total, total_EAD = granularity_adjustment(
    df, K, R_val, delta, gamma
)

df["LGD_Variance"] = Var_LGD
df["LGD_Factor_C"] = C
df["ExposureShare_s"] = s
df["GranularityAdjustment_GA"] = GA
df["Portfolio_K_Star"] = K_star_total

GA_capital, GA_RWA = calc_GA_capital_RWA(total_GA, total_EAD, charge_pct)

# -----------------------------------------------------------
# Display Results
# -----------------------------------------------------------

st.subheader("📊 Customer-Level Detailed Output")
st.dataframe(df)

summary = pd.DataFrame({
    "Total EAD": [total_EAD],
    "Total Portfolio Granularity Adjustment": [total_GA],
    "GA Capital Charge": [GA_capital],
    "GA RWA (Capital ÷ Charge %)": [GA_RWA],
    "Charge-to-EAD %": [charge_pct],
    "K-Star (Σ sᵢ Kᵢ)": [K_star_total]
})

st.subheader("📘 Portfolio GA Summary")
st.dataframe(summary)

# -----------------------------------------------------------
# Download Excel Output
# -----------------------------------------------------------

output = io.BytesIO()
with pd.ExcelWriter(output, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Customer_Metrics")
    summary.to_excel(writer, index=False, sheet_name="GA_Summary")

st.download_button(
    "📥 Download Excel Results",
    data=output.getvalue(),
    file_name="Granularity_Adjustment_Output.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
