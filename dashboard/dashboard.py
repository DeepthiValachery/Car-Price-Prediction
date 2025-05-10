import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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

    # For Future Trends
    yearly_prices = filtered_df.groupby('Year')['Price'].mean().reset_index()

    X = yearly_prices[['Year']]
    y = yearly_prices['Price']
    model = LinearRegression()
    model.fit(X, y)

    # Predict next 6 years
    future_years = pd.DataFrame({'Year': np.arange(filtered_df['Year'].max()+1, filtered_df['Year'].max()+7)})
    future_prices = model.predict(future_years)

    # Combine actual and predicted
    future_df = future_years.copy()
    future_df['Price'] = future_prices
    future_df['Type'] = 'Predicted'
    yearly_prices['Type'] = 'Actual'
    combined_df = pd.concat([yearly_prices, future_df])

    st.write("") 
    st.title("Trends from the Filtered Listings")

    #  =========== Filtered Visuals =============

    st.write("")
    st.subheader("Price vs Mileage")

    smoothed_df = filtered_df[["Mileage", "Price"]].sort_values("Mileage")

    # Fit Polynomial Regression (2nd degree for smooth curve)
    X1 = smoothed_df["Mileage"].values.reshape(-1, 1)
    y1 = smoothed_df["Price"].values
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X1)
    model = LinearRegression().fit(X_poly, y1)
    y_pred = model.predict(X_poly)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=smoothed_df["Mileage"], y=smoothed_df["Price"],
                            mode='markers', name='Actual Data', opacity=0.4))
    fig.add_trace(go.Scatter(x=smoothed_df["Mileage"], y=y_pred,
                            mode='lines', name='Price vs Mileage', line=dict(color='white', width=3)))

    fig.update_layout(title="The graph shows that as the mileage of a vehicle increases, the price decreases",
                    xaxis_title="Mileage",
                    yaxis_title="Price",
                    height=400)

    st.plotly_chart(fig, use_container_width=True)
 

    st.subheader("Average Price of Vehicle by Year")
    avg_by_year = (
        filtered_df.groupby("Year")["Price"]
        .mean()
        .reset_index()
        .sort_values("Year")
    )
    st.line_chart(avg_by_year, x="Year", y="Price")

    st.subheader("Manufacturing Year vs Price")
    st.scatter_chart(filtered_df[["Year", "Price"]])

    st.subheader("Car Price Trends and Predictions")

    st.write("The graph below shows the predicted trend for car prices for the next six years") 

    line_chart = alt.Chart(combined_df).mark_line(point=True).encode(
        x='Year:O',
        y='Price:Q',
        color='Type:N'
    ).properties(width=700)

    st.altair_chart(line_chart, use_container_width=True)

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
