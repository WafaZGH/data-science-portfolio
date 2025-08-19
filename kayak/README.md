# Plan Your Trip with Kayak - Data Project

This project was part of my data science training.  
The goal was to help travelers choose the best destinations in France by combining **7-day weather forecasts** and **hotel availability**, and visualizing the results on an interactive map.



## The Challenge
We worked with two datasets stored in PostgreSQL:

- **weather_7day** → daily min/max temperatures for French cities  
- **all_cities_hotels** → hotel names, locations, and metadata  

Objectives:
1. Identify the **Top-5 French cities** with an average temperature below 30 °C  
2. Select the **Top-20 hotels** for each of those cities  
3. Display everything on a single interactive Plotly map  



## Context
The pipeline combines:
- **API data** from [Open-Meteo](https://api.open-meteo.com/v1/forecast) (7-day forecasts, no API key required)  
- **Web scraping** from Booking.com (hotel listings)  

Data was stored in PostgreSQL, then transformed and visualized to create a simple travel planning guide.



## My Approach

### 1. Data Collection
- **Weather (Open-Meteo API):** Queried daily variables for each city using latitude/longitude, concatenated results, and saved to PostgreSQL (`kayak.weather_7day`).  
- **Hotels (Booking.com scraping):** Collected hotel cards from search results with Requests + BeautifulSoup, then geocoded and stored in PostgreSQL (`kayak.all_cities_hotels`).  

### 2. Data Transformation
- Computed average temperatures per city from `weather_7day`  
- Filtered only French cities (lat/lon bounds)  
- Selected **Top-5 coolest cities (<30 °C)**  
- Retrieved **Top-20 hotels** for each selected city  

### 3. Visualization
- Built an interactive **Plotly Scattermapbox**:  
  - 🔵 Blue markers → Top-5 cities  
  - 🔴 Red markers → Top-20 hotels (hover shows hotel name)  



## What I Learned
- Building a small ETL pipeline (API ingestion + scraping → SQL)  
- Combining weather and hotel data into a travel recommender  
- Creating clean, interactive maps with Plotly Mapbox  
- Handling scraping nuances (headers, polite delays, etc.)  



## Files
- `Kayak.ipynb` → main notebook  
- `data/7_day_weather_forecast.csv` → weather data export  
- `data/all_cities_hotels.csv` → hotels export  
- `README.md` → this file  



## Next Steps

-Right now, the analysis only looks at the **7-day weather forecast**. That’s useful for last-minute trips, but not very practical for travelers who book **3 to 6 months in advance**.  

To make this tool more reliable, the next step would be to include **monthly averages** and **long-term climate trends**. With that, we could:  
- Show the **best months** to visit each destination  
- Compare places based on **seasonal weather patterns**  
- Provide a more trustworthy guide for advance travel planning  






