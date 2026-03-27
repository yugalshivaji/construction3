# Delhi NCR Construction Tracking Dashboard

A comprehensive dashboard for monitoring construction activities in Delhi NCR using satellite imagery (Sentinel-2), OpenStreetMap data, and custom GeoJSON uploads.

## Features

- **Real-time Change Detection**: Monitor construction activities by comparing satellite imagery from different time periods
- **Construction Heatmap**: Visualize construction density with smoothed heatmaps
- **OSM Data Integration**: Fetch and display construction sites from OpenStreetMap
- **GeoJSON Upload**: Upload custom construction site data
- **Street View Integration**: Explore locations with Google Street View
- **Interactive Filters**: Filter sites by type (Buildings, Roads, Land Development)
- **Dark Mode UI**: Praan AI inspired green-accented dark theme

## Tech Stack

### Backend
- Flask (Python)
- Google Earth Engine API
- Flask-CORS
- Gunicorn (production)

### Frontend
- HTML5/CSS3/JavaScript
- Google Maps JavaScript API
- Overpass API (OSM data)

## Local Development Setup

### Prerequisites
- Python 3.8+
- Google Earth Engine account
- Google Maps API key

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/delhi-construction-tracker.git
cd delhi-construction-tracker
