
import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import io
import requests

# --- Load Data ---
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # FEMA ZIP URL
        url = "https://hazards.fema.gov/nri/Content/StaticDocuments/DataDownload/NRI_Table_Counties/NRI_Table_Counties.zip"
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                county_df = pd.read_csv(f)

        # Add Region mapping
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
    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    regions = county_df['REGION'].dropna().unique()
    selected_region = st.sidebar.multiselect("Select Region(s)", sorted(regions), default=regions)

    states = county_df['STATE'].unique()
    selected_state = st.sidebar.multiselect("Select State(s)", sorted(states), default=states)

    counties = county_df['COUNTY'].unique()
    selected_county = st.sidebar.multiselect("Select County(s)", sorted(counties))

    # --- Filter Data ---
    filtered_df = county_df[
        county_df['REGION'].isin(selected_region) &
        county_df['STATE'].isin(selected_state)
    ]
    if selected_county:
        filtered_df = filtered_df[filtered_df['COUNTY'].isin(selected_county)]

    # --- Dashboard Content ---
    st.title("National Risk Index Interactive Dashboard")
    st.markdown("Explore Risk, Resilience & Loss Metrics by Region, State, and County")

    st.write(f"Filtered data contains **{len(filtered_df)} rows**.")

    # --- Map ---
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

    # --- Barplot ---
    st.subheader("Average Risk Score by State")
    avg_risk = filtered_df.groupby('STATE')['RISK_SCORE'].mean().sort_values(ascending=False).reset_index()
    fig_bar = px.bar(avg_risk, x='STATE', y='RISK_SCORE', color='RISK_SCORE',
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Scatterplot ---
    st.subheader("Risk Score vs Population")
    fig_scatter = px.scatter(
        filtered_df, x='POPULATION', y='RISK_SCORE', color='STATE',
        hover_data=['COUNTY'], title="Risk Score vs Population"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Download Option ---
    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "filtered_data.csv", "text/csv")

    # --- Footer ---
    st.markdown("---")
    st.caption("National Risk Index Interactive Dashboard | QNT980 Project")
else:
    st.error("Dataset could not be loaded. Please check your internet connection.")
