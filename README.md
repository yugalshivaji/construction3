
# 🏗️ Delhi NCR Construction Tracking Dashboard

## Real-Time Construction Monitoring & Satellite Intelligence Platform

<p align="center">

![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Google Earth Engine](https://img.shields.io/badge/Google%20Earth%20Engine-34A853?style=for-the-badge&logo=google&logoColor=white)
![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-7EBC6F?style=for-the-badge&logo=openstreetmap&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</p>

---

# 🌍 Overview

**Delhi NCR Construction Tracking Dashboard** is a cutting-edge web platform for monitoring construction activities across the Delhi National Capital Region. Leveraging satellite imagery from Sentinel-2, OpenStreetMap data, and custom GeoJSON uploads, the platform provides real-time construction intelligence through interactive visualizations and advanced change detection algorithms.

Built with a modern dark-mode interface inspired by environmental intelligence platforms, the dashboard empowers urban planners, researchers, real estate developers, and public authorities to track construction patterns, analyze urban growth, and make data-driven decisions.

Whether monitoring infrastructure development, tracking land-use changes, or analyzing construction density, this platform transforms complex satellite and geospatial data into actionable insights.

---

# 🎯 Vision

Rapid urbanization in Delhi NCR demands innovative approaches to construction monitoring and urban planning.

Traditional methods are often slow, fragmented, and inaccessible.

This platform addresses these challenges by providing:

* 🛰️ Satellite-Based Construction Detection
* 📊 Real-Time Change Analysis
* 🗺️ Interactive Geospatial Visualizations
* 🔍 Smart Search & Filtering
* 📡 Multi-Source Data Integration
* 🌐 Accessible Construction Intelligence

The goal is to democratize access to construction monitoring data and enable smarter urban development.

---

# ✨ Features

## 🛰️ Satellite-Based Change Detection

Monitor construction activities by comparing Sentinel-2 satellite imagery from different time periods.

Key capabilities include:

* Automated Change Detection Algorithm
* NDVI / NDBI / NDWI Index Calculations
* Custom Time Period Selection
* Adjustable Change Threshold
* Visual Change Score Legend

---

## 🌡️ Construction Heatmap

Visualize construction density across Delhi NCR using smoothed Gaussian heatmaps.

Features include:

* Intensity-Based Heat Mapping
* Adjustable Blur Radius
* Percentile-Stretched Visual Contrast
* Density Analysis

---

## 🏙️ OSM Data Integration

Fetch and display construction sites from OpenStreetMap in real-time.

Supports:

* Building Projects
* Road & Highway Works
* Land Development Sites
* Live Data Updates

---

## 🗺️ GeoJSON Upload & Visualization

Upload custom construction site data in GeoJSON format.

Supports:

* File Upload (Drag & Drop)
* URL Loading
* Sample Data Preview
* Local Layer Styling

---

## 🔭 Google Street View Integration

Explore construction sites from street level.

Features:

* One-Click Street View Activation
* Click-to-Teleport Navigation
* Seamless Map Integration

---

## 🔍 Smart Search & Location

Find construction sites across Delhi NCR.

Includes:

* Place Search with Auto-Suggest
* Current Location Detection
* Dynamic Map Centering

---

## 🏛️ Interactive Filtering

Filter construction sites by category:

* 🏢 Building Projects
* 🛣️ Road & Highway Works
* 🏗️ Land Development
* 📍 GeoJSON Sites

---

## 📋 Active Sites List

Browse construction sites in a structured list view.

Features:

* All / OSM / GeoJSON Tabs
* Site Details Preview
* Click-to-Focus Navigation
* Visual Badge Indicators

---

## 📡 Sentinel-2 True Colour Layer

Display high-resolution true-colour satellite imagery.

Features:

* Custom Date Range Selection
* Toggle Visibility
* Cloud-Filtered Composites
* Seamless Overlay

---

## 🎨 Dark Mode Interface

Praan AI inspired green-accented dark theme.

Features:

* Optimized for Night-Time Use
* High Contrast Design
* Eco-Friendly Aesthetic

---

# 🏗 System Architecture

```text
                    Client Browser
                         │
                         ▼
           HTML + JavaScript Dashboard
                         │
              Google Maps API / OSM
                         │
                         ▼
              Flask Backend (Python)
                         │
                         ▼
           Google Earth Engine API
                         │
                         ▼
            Sentinel-2 Satellite Data
                         │
                         ▼
        Change Detection / Heatmap Engine
                         │
                         ▼
         Interactive Geospatial Dashboard
```

---

# 📂 Project Structure

```text
yugalshivaji-construction3/
│
├── index.html
├── delhi_ncr_construction_sites_v2.geojson
├── render.yaml
├── README.md
└── backend/
    ├── app.py
    └── requirements.txt
```

---

# 🛠 Technology Stack

## Frontend

* HTML5
* Tailwind CSS
* JavaScript (ES6)
* Google Maps JavaScript API
* OpenStreetMap Overpass API
* osmtogeojson Library

---

## Backend

* Flask 3.0.3
* Google Earth Engine API
* Flask-CORS
* Gunicorn (Production)

---

## APIs & Services

* Google Earth Engine
* Sentinel-2 Satellite Data
* Google Maps API
* OpenStreetMap Overpass API
* Google Street View API
* Browser Geolocation API

---

# 📊 Dashboard Components

The dashboard provides comprehensive construction intelligence through multiple interactive components.

### Status & Statistics

* Real-Time Data Status
* OSM Sites Count
* GeoJSON Sites Count

---

### Change Detection Panel

* Baseline Period Selection (T1)
* Recent Period Selection (T2)
* Threshold Adjustment
* Kernel Radius Control (Heatmap)
* Change Score Legend
* Area Statistics Display

---

### Satellite Layer Controls

* Sentinel-2 True Colour
* Date Range Selection
* Toggle Visibility

---

### GeoJSON Data Loader

* File Upload with Drag & Drop
* URL Loading
* Sample Data Loader
* Layer Clear Option

---

### Street View Integration

* One-Click Activation
* Interactive Navigation
* Map Click-to-Teleport

---

### Filter Controls

* Building Projects Filter
* Road & Highway Filter
* Land Development Filter
* GeoJSON Sites Filter

---

### Active Sites List

* All / OSM / GeoJSON Tabs
* Site Cards with Details
* Click-to-Focus Navigation
* Visual Badges

---

# 🔄 Application Workflow

```text
User Opens Dashboard
        │
        ▼
Initialize Map (Delhi NCR)
        │
        ▼
Load Sentinel-2 Satellite Layer
        │
        ▼
Fetch OSM Construction Data
        │
        ▼
Load GeoJSON (if uploaded)
        │
        ▼
Interactive Map Display
        │
        ▼
Search • Filter • Focus
        │
        ▼
Change Detection / Heatmap Analysis
        │
        ▼
Street View Exploration
```

---

# 🌐 Data Sources

The platform integrates data from multiple authoritative sources:

| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| Sentinel-2 | Satellite Imagery | 5 Days |
| Google Earth Engine | Processed Imagery | Real-Time |
| OpenStreetMap | Construction Sites | Community-Driven |
| GeoJSON Files | Custom Site Data | User-Defined |

---

# 🖥️ Backend API Endpoints

## Health Check
```
GET /api/health
```

## Sentinel-2 True Colour
```
GET /api/gee-construction?date_start={date}&date_end={date}
```

## Change Detection
```
GET /api/gee-change-detection?t1_start={date}&t1_end={date}&t2_start={date}&t2_end={date}&threshold={float}
```

## Construction Heatmap
```
GET /api/gee-heatmap?t1_start={date}&t1_end={date}&t2_start={date}&t2_end={date}&kernel_radius={int}
```

## Area Statistics
```
GET /api/gee-stats?minLng={float}&minLat={float}&maxLng={float}&maxLat={float}&t1_start={date}&t1_end={date}&t2_start={date}&t2_end={date}
```

---

# ⚙ Installation

## Prerequisites

* Python 3.8+
* Google Earth Engine Account
* Google Cloud Project
* Google Maps API Key

---

## Clone Repository

```bash
git clone https://github.com/YugalOfficial/yugalshivaji-construction3.git

cd yugalshivaji-construction3
```

---

## Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the backend directory:

```env
GEE_PRIVATE_KEY=<your-service-account-private-key>
GEE_CLIENT_EMAIL=<your-service-account-email>
GEE_PROJECT_ID=<your-project-id>
```

---

## Run Backend Locally

```bash
python app.py
```

Backend will run at: `http://localhost:5000`

---

## Frontend Setup

Simply open `index.html` in a browser, or serve using a local server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js (serve)
npx serve
```

Visit: `http://localhost:8000`

---

# 🚀 Deployment

## Backend Deployment (Render)

```yaml
# render.yaml
services:
  - type: web
    name: delhi-construction-backend
    runtime: python
    plan: free
    region: frankfurt
    buildCommand: cd backend && pip install -r requirements.txt
    startCommand: cd backend && gunicorn app:app
    envVars:
      - key: GEE_PRIVATE_KEY
        sync: false
      - key: GEE_CLIENT_EMAIL
        sync: false
      - key: GEE_PROJECT_ID
        sync: false
```

---

## Frontend Deployment

Deploy on:

* GitHub Pages
* Netlify
* Vercel
* Firebase Hosting
* Cloudflare Pages

---

# 🔒 Error Handling

The application gracefully manages:

* Network Failures
* API Connection Errors
* Earth Engine Initialization Failures
* Empty Search Results
* Invalid GeoJSON Format
* Browser Geolocation Denial
* Missing Internet Connectivity
* Unsupported Browser Features

Users receive informative messages to ensure a smooth experience.

---

# 🌟 Future Enhancements

* 📊 Historical Construction Analytics
* 🤖 AI-Powered Construction Prediction
* 🚁 Drone Imagery Integration
* 📱 Progressive Web App (PWA)
* 🔔 Real-Time Alerts & Notifications
* 🌍 Multi-City Expansion
* 📈 Construction Volume Estimation
* 🏗️ Project Timeline Visualization
* 🗺️ 3D Terrain Visualization
* 📅 Construction Permit Integration
* 🤝 Citizen Reporting Module
* 🌐 International Language Support

---

# 💡 Project Highlights

* 🛰️ Satellite-Based Construction Detection
* 📊 Real-Time Change Analysis
* 🌡️ Construction Density Heatmap
* 🗺️ Interactive Geospatial Dashboard
* 📡 Sentinel-2 True Colour Integration
* 🏙️ OpenStreetMap Data Integration
* 📂 GeoJSON File Upload Support
* 🔭 Google Street View Integration
* 🎨 Praan AI Inspired Dark Theme
* 🌐 Flask Backend with Google Earth Engine
* 📱 Fully Responsive Design
* ☁️ Cloud-Ready Deployment

---

# 🤝 Contributing

Contributions are welcome and greatly appreciated.

1. Fork the repository.

2. Create a feature branch.

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes.

```bash
git commit -m "Add New Feature"
```

4. Push your branch.

```bash
git push origin feature/NewFeature
```

5. Submit a Pull Request.

---

# 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project with appropriate attribution.

---

# 👨‍💻 Author

## **Yugal**

**AI Developer • Full Stack Developer • Software Engineer**

Passionate about building AI-powered applications, geospatial intelligence platforms, and scalable web solutions that create meaningful real-world impact through technology.

---

# 🙏 Acknowledgements

Special thanks to the technologies and services that power this project:

* Google Earth Engine
* Sentinel-2 Satellite Program
* OpenStreetMap Community
* Google Maps Platform
* Tailwind CSS
* Flask Framework
* Render Platform

---

# ⭐ Support the Project

If you found this project useful, please consider giving it a **⭐ Star** on GitHub.

Your support motivates the development of more innovative open-source applications focused on geospatial intelligence, urban planning, and digital public services.

---

<p align="center">

## ❤️ Empowering Urban Intelligence Through Satellite Technology

### Built with dedication and innovation by **Yugal**

**"Satellite data holds the key to smarter, more sustainable cities."**

</p>
