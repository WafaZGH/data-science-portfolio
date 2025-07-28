# Uber Pickup Clustering Project

This project identifies high-demand Uber pickup zones in New York City using clustering techniques.  
By analyzing ride data from April to September 2014, we help optimize driver positioning, reduce passenger wait times, and uncover hidden demand patterns.



##  Objective

To detect spatial and temporal patterns in Uber ride data and define optimal pickup zones using clustering methods like **KMeans** and **DBSCAN**.



## Dataset

- **Source:** Uber pickup logs (April–September 2014)
- **Features:** `Date/Time`, `Latitude`, `Longitude`
- **Size:** ~4.5 million rows across 6 monthly CSV files



##  Project Workflow

1. **Data Merging**  
   Combined all six monthly CSV files into a single DataFrame

2. **Preprocessing**
   - Converted `Date/Time` to datetime objects
   - Extracted `hour`, `day`, `weekday` features
   - Dropped null coordinates & sorted by time

3. **Exploratory Data Analysis**
   - Pickup scatter maps
   - Bar charts of pickups per hour/day
   - Heatmap: hours vs days

4. **Clustering Approaches**

   ** KMeans**
   - Tested various `k` values
   - Used **Elbow Method** and **Silhouette Score** to choose `k = 6, 9, 10`
   - Visualized clusters on Plotly map

   ** DBSCAN**
   - Detected organic, density-based clusters
   - Flagged noise and sparse pickup areas

5. **Time-Based Visualization**
   - Interactive hourly density maps (Plotly)
   - Showed how hot zones shift over the day



## Key Insights

- Peak demand: **3 PM to 10 PM**, especially Thursdays  
- **Manhattan (Midtown & Downtown)** are consistently high-traffic zones  
- Weekends show broader pickup distribution across boroughs  
- **DBSCAN** is effective for noise filtering and finding irregular clusters



##  Tools & Libraries

- **Python** (Pandas, NumPy)
- **scikit-learn** (KMeans, DBSCAN)
- **Plotly** for interactive maps
- **Seaborn**, **Matplotlib** for visualizations



📁 _Notebook: `uber-pickups.ipynb` 
📂 _Data files: 6 monthly CSVs



