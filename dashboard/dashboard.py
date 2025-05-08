import streamlit as st
import pandas as pd
import plotly.express as px

# Load and save the CSV Data
@st.cache_data 
def load_data():
    df = pd.read_csv("dashboard/Cars.csv") 
    df.dropna(subset=["Price", "Mileage", "Year", "Make", "State"], inplace=True)
    return df

df = load_data()

st.title("Car Pricing Advisor")

# Sidebar
st.sidebar.header("Filter Listings")
st.markdown("Use the Filter to explore car listings and identify competitive price ranges.")

states = df["State"].unique()
makes = df["Make"].unique()
years = df["Year"].astype(int)

selected_state = st.sidebar.selectbox("Select State", sorted(states))
selected_make = st.sidebar.selectbox("Select Car Model", sorted(makes))
year_range = st.sidebar.slider("Manufacturing Year", int(years.min()), int(years.max()), (2000, 2024))
mileage_max = st.sidebar.slider("Maximum Mileage", int(df["Mileage"].min()), int(df["Mileage"].max()), int(df["Mileage"].max()))

# Filter the Data 
filtered_df = df[
    (df["State"] == selected_state) &
    (df["Make"] == selected_make) &
    (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1]) &
    (df["Mileage"] <= mileage_max)
]

st.subheader(f"Matching Listings for {selected_make} in {selected_state}")

# Display Filtered Car Listings
if filtered_df.empty:
    st.warning("No listings match the selected criteria. Try relaxing your filters.")
else:
    st.write(f"Filtered to display listings between {year_range[0]} and {year_range[1]} with mileage under {mileage_max:,}.")
    st.dataframe(filtered_df[["Price", "Year", "Make", "Mileage", "Drivetrain", "Fuel Type", "Convenience Features", "Entertainment Features", "Exterior Features", "Safety Features"]])

    # Average Price (Suggestions)
    avg_price = filtered_df["Price"].mean()
    min_price = filtered_df["Price"].min()
    max_price = filtered_df["Price"].max()

    st.success(f"Suggested Price Range: ${min_price:,.0f} – ${max_price:,.0f}")
    st.info(f"Average Price: ${avg_price:,.0f}")

    st.write("") 
    st.title("Trends from the Filtered Listings")

    #  =========== Filtered Visuals for the Dealer =============
    
    st.write("") 
    st.subheader("Price vs Mileage")
    st.line_chart(filtered_df[["Mileage", "Price"]])

    st.subheader("Average Price by Year")
    avg_by_year = (
        filtered_df.groupby("Year")["Price"]
        .mean()
        .reset_index()
        .sort_values("Year")
    )
    st.line_chart(avg_by_year, x="Year", y="Price")

    st.subheader("Year vs Price")
    st.scatter_chart(filtered_df[["Year", "Price"]])

    # =========== General Visuals =============

    st.write("") 
    st.title("General Car Trends")

    st.subheader("Most Common Car Makes")
    top_makes = df["Make"].value_counts().head(10)
    st.bar_chart(top_makes)

    st.subheader("Listings by State")
    state_counts = df["State"].value_counts()
    st.bar_chart(state_counts)

    st.subheader("Exterior Color Popularity")
    color_counts = df["Exterior Color"].value_counts().head(10)
    st.bar_chart(color_counts)
