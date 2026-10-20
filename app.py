# ==============================================================================
# STREAMLIT REAL ESTATE ANALYTICS DASHBOARD (SELF-CONTAINED)
# Parcl Co. Limited | Unified Mentor AI Buyer Intelligence
# ==============================================================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Parcl Real Estate Intelligence",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Parcl Co. Limited | Buyer Segmentation Intelligence")
st.caption("AI-Driven Real Estate Buyer Profiling & Investment Analytics")
st.divider()

# Load dataset with automatic fallback if csv is not yet exported
@st.cache_data
def load_data():
    csv_file = "segmented_real_estate_clients.csv"
    if not os.path.exists(csv_file):
        # Auto-generate if missing
        clients = pd.read_csv("clients.csv")
        properties = pd.read_csv("properties.csv")
        properties['sale_price_clean'] = properties['sale_price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)
        portfolio = properties.groupby('client_ref').agg(
            total_investment=('sale_price_clean', 'sum'),
            avg_property_price=('sale_price_clean', 'mean'),
            properties_purchased=('listing_id', 'count')
        ).reset_index()
        df_merged = pd.merge(clients, portfolio, left_on='client_id', right_on='client_ref', how='left')
        df_merged['dob'] = pd.to_datetime(df_merged['date_of_birth'], format='mixed', errors='coerce')
        df_merged['age'] = (2026 - df_merged['dob'].dt.year).fillna(55)
        df_merged['properties_purchased'] = df_merged['properties_purchased'].fillna(0)
        df_merged['total_investment'] = df_merged['total_investment'].fillna(0)
        df_merged['avg_property_price'] = df_merged['avg_property_price'].fillna(0)
        df_merged['satisfaction_score'] = df_merged['satisfaction_score'].fillna(3)
        df_merged['loan_applied_num'] = df_merged['loan_applied'].map({'Yes': 1, 'No': 0}).fillna(0)

        num_cols = ['age', 'satisfaction_score', 'total_investment', 'avg_property_price', 'properties_purchased', 'loan_applied_num']
        cat_cols = ['client_type', 'country', 'region', 'acquisition_purpose', 'referral_channel']

        scaler = StandardScaler()
        encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        X = np.hstack([scaler.fit_transform(df_merged[num_cols]), encoder.fit_transform(df_merged[cat_cols])])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_merged['cluster_id'] = kmeans.fit_predict(X)
        persona_dict = {
            0: "C1: Global Investors",
            1: "C2: First-Time Buyers",
            2: "C3: Corporate Buyers",
            3: "C4: Luxury Investors"
        }
        df_merged['buyer_segment'] = df_merged['cluster_id'].map(persona_dict)
        coords = PCA(n_components=2).fit_transform(X)
        df_merged['pca_1'] = coords[:, 0]
        df_merged['pca_2'] = coords[:, 1]
        df_merged.to_csv(csv_file, index=False)
        return df_merged
    return pd.read_csv(csv_file)

df = load_data()

# ------------------------------------------------------------------------------
# USER CONTROLS: SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.header("🔍 User Controls")

countries = sorted(df['country'].dropna().unique())
selected_country = st.sidebar.multiselect("Country", options=countries, default=countries)

valid_regions = sorted(df[df['country'].isin(selected_country)]['region'].dropna().unique())
selected_region = st.sidebar.multiselect("Region", options=valid_regions, default=valid_regions)

purposes = sorted(df['acquisition_purpose'].dropna().unique())
selected_purpose = st.sidebar.multiselect("Acquisition Purpose", options=purposes, default=purposes)

client_types = sorted(df['client_type'].dropna().unique())
selected_client_type = st.sidebar.multiselect("Client Type", options=client_types, default=client_types)

filtered_df = df[
    (df['country'].isin(selected_country)) &
    (df['region'].isin(selected_region)) &
    (df['acquisition_purpose'].isin(selected_purpose)) &
    (df['client_type'].isin(selected_client_type))
]

if filtered_df.empty:
    st.warning("No client records match the selected filters. Please expand your criteria.")
    st.stop()

# ------------------------------------------------------------------------------
# KPI METRICS
# ------------------------------------------------------------------------------
st.markdown("""
    <style>
    /* Metric Card text size and overflow fix */
    div[data-testid="stMetricValue"] > div {
        font-size: 19px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #4b5563 !important;
    }
    </style>
""", unsafe_allow_html=True)

total_val = filtered_df['total_investment'].sum()
avg_val = filtered_df['total_investment'].mean()
satisfaction_val = filtered_df['satisfaction_score'].mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Active Buyers", f"{len(filtered_df):,}")
m2.metric("Total Portfolio Value", f"${total_val:,.0f}")
m3.metric("Avg Investment / Buyer", f"${avg_val:,.0f}")
m4.metric("Avg Satisfaction Rating", f"{satisfaction_val:.2f} / 5.0")

st.divider()
# ------------------------------------------------------------------------------
# CORE DASHBOARD MODULES
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Buyer Segmentation Overview",
    "📈 Investor Behavior Dashboard",
    "🌍 Geographic Buyer Analysis",
    "📋 Segment Insights Panel"
])

with tab1:
    st.subheader("Buyer Segmentation Overview")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        seg_dist = filtered_df['buyer_segment'].value_counts().reset_index()
        seg_dist.columns = ['Segment', 'Count']
        fig_pie = px.pie(seg_dist, values='Count', names='Segment', title="Cluster Distribution Ratio", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_pca = px.scatter(
            filtered_df, x='pca_1', y='pca_2', color='buyer_segment',
            hover_data=['client_id', 'total_investment', 'age'],
            title="2D PCA Latent Space Distribution"
        )
        st.plotly_chart(fig_pca, use_container_width=True)

with tab2:
    st.subheader("Investor Behavior Dashboard")
    c1, c2 = st.columns(2)
    with c1:
        fig_box = px.box(filtered_df, x='buyer_segment', y='total_investment', color='buyer_segment', title="Investment Capital Distribution")
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with c2:
        loan_agg = filtered_df.groupby(['buyer_segment', 'loan_applied']).size().reset_index(name='Clients')
        fig_loan = px.bar(loan_agg, x='buyer_segment', y='Clients', color='loan_applied', barmode='group', title="Financing & Loan Patterns")
        st.plotly_chart(fig_loan, use_container_width=True)

with tab3:
    st.subheader("Geographic Buyer Analysis")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        geo_agg = filtered_df.groupby(['country', 'buyer_segment']).size().reset_index(name='Clients')
        fig_country = px.bar(geo_agg, x='country', y='Clients', color='buyer_segment', title="Buyer Segments Across Countries", barmode='stack')
        st.plotly_chart(fig_country, use_container_width=True)
    with c2:
        top_reg = filtered_df['region'].value_counts().head(10).reset_index()
        top_reg.columns = ['Region', 'Buyers']
        fig_region = px.bar(top_reg, x='Buyers', y='Region', orientation='h', title="Top 10 Active Regions", color='Buyers')
        fig_region.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_region, use_container_width=True)

with tab4:
    st.subheader("Segment Insights Panel")
    summary = filtered_df.groupby('buyer_segment').agg(
        Total_Buyers=('client_id', 'count'),
        Avg_Age=('age', 'mean'),
        Avg_Investment=('total_investment', 'mean'),
        Avg_Property_Price=('avg_property_price', 'mean'),
        Avg_Units=('properties_purchased', 'mean'),
        Avg_Satisfaction=('satisfaction_score', 'mean'),
        Loan_Dependency_Rate=('loan_applied_num', lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()

    summary['Avg_Age'] = summary['Avg_Age'].round(1)
    summary['Avg_Investment'] = summary['Avg_Investment'].apply(lambda x: f"${x:,.2f}")
    summary['Avg_Property_Price'] = summary['Avg_Property_Price'].apply(lambda x: f"${x:,.2f}")
    summary['Avg_Units'] = summary['Avg_Units'].round(2)
    summary['Avg_Satisfaction'] = summary['Avg_Satisfaction'].round(2)
    st.dataframe(summary, use_container_width=True)
    st.divider()

    st.markdown("#### Recommended Buyer Personas & Strategic Definitions")
    colA, colB = st.columns(2)
    with colA:
        st.info("**C1: Global Investors** — High income, investment-focused purchases.")
        st.info("**C2: First-Time Buyers** — Younger demographic, predominantly loan-dependent.")
    with colB:
        st.info("**C3: Corporate Buyers** — Companies purchasing multiple property portfolios.")
        st.info("**C4: Luxury Investors** — High satisfaction scores with premium acquisitions.")
