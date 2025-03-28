
import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import io
import requests
import statsmodels.api as sm
import numpy as np

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_data():
    try:
        url = "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties/NRI_Table_Counties.zip"
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                county_df = pd.read_csv(f)

        region_map = {
            'Connecticut': 'Northeast', 'Maine': 'Northeast', 'Massachusetts': 'Northeast', 'New Hampshire': 'Northeast',
            'Rhode Island': 'Northeast', 'Vermont': 'Northeast', 'New Jersey': 'Northeast', 'New York': 'Northeast',
            'Pennsylvania': 'Northeast', 'Illinois': 'Midwest', 'Indiana': 'Midwest', 'Michigan': 'Midwest',
            'Ohio': 'Midwest', 'Wisconsin': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Minnesota': 'Midwest',
            'Missouri': 'Midwest', 'Nebraska': 'Midwest', 'North Dakota': 'Midwest', 'South Dakota': 'Midwest',
            'Delaware': 'South', 'Florida': 'South', 'Georgia': 'South', 'Maryland': 'South', 'North Carolina': 'South',
            'South Carolina': 'South', 'Virginia': 'South', 'District of Columbia': 'South', 'West Virginia': 'South',
            'Alabama': 'South', 'Kentucky': 'South', 'Mississippi': 'South', 'Tennessee': 'South', 'Arkansas': 'South',
            'Louisiana': 'South', 'Oklahoma': 'South', 'Texas': 'South', 'Arizona': 'West', 'Colorado': 'West',
            'Idaho': 'West', 'Montana': 'West', 'Nevada': 'West', 'New Mexico': 'West', 'Utah': 'West', 'Wyoming': 'West',
            'Alaska': 'West', 'California': 'West', 'Hawaii': 'West', 'Oregon': 'West', 'Washington': 'West'
        }
        county_df['REGION'] = county_df['STATE'].map(region_map)

        return county_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

county_df = load_data()

if not county_df.empty:
    st.sidebar.header("Filters")
    regions = county_df['REGION'].dropna().unique()
    selected_region = st.sidebar.multiselect("Select Region(s)", sorted(regions), default=regions)
    states = county_df[county_df['REGION'].isin(selected_region)]['STATE'].unique()
    selected_state = st.sidebar.multiselect("Select State(s)", sorted(states), default=states)
    counties = county_df[county_df['STATE'].isin(selected_state)]['COUNTY'].unique()
    selected_county = st.sidebar.multiselect("Select County(s)", sorted(counties))

    filtered_df = county_df[
        county_df['REGION'].isin(selected_region) &
        county_df['STATE'].isin(selected_state)
    ]
    if selected_county:
        filtered_df = filtered_df[filtered_df['COUNTY'].isin(selected_county)]

    st.title("National Risk Index Interactive Dashboard")
    st.markdown("Explore Risk, Resilience & Loss Metrics by Region, State, and County")
    st.write(f"Filtered data contains **{len(filtered_df)} rows**.")

    st.subheader("Risk Score by County")
    fig = px.choropleth(
        filtered_df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations=filtered_df['STCOFIPS'].astype(str).str.zfill(5),
        color='RISK_SCORE',
        scope="usa",
        hover_data=['STATE', 'COUNTY', 'RISK_SCORE'],
        color_continuous_scale="OrRd",
        labels={'RISK_SCORE': 'Risk Score'}
    )
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Risk Score by State")
    avg_risk = filtered_df.groupby('STATE')['RISK_SCORE'].mean().sort_values(ascending=False).reset_index()
    fig_bar = px.bar(avg_risk, x='STATE', y='RISK_SCORE', color='RISK_SCORE',
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Risk Score vs Population")
    fig_scatter = px.scatter(
        filtered_df, x='POPULATION', y='RISK_SCORE', color='STATE',
        hover_data=['COUNTY'], title="Risk Score vs Population"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    if st.checkbox("Run Simple Regression: Predict EAL_VALT"):
        reg_cols = ['EAL_VALT', 'RISK_SCORE', 'SOVI_SCORE', 'RESL_SCORE', 'POPULATION']
        reg_df = filtered_df[reg_cols].dropna().copy()
        if not reg_df.empty:
            reg_df['log_EAL_VALT'] = np.log1p(reg_df['EAL_VALT'])
            X = reg_df[['RISK_SCORE', 'SOVI_SCORE', 'RESL_SCORE', 'POPULATION']]
            y = reg_df['log_EAL_VALT']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            st.write("**Regression Results:**")
            st.text(model.summary())

    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

    st.markdown("---")
    st.caption("National Risk Index Interactive Dashboard | QNT980 Project")
else:
    st.error("Dataset could not be loaded. Please check your internet connection.")
