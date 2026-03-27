from flask import Flask, jsonify, request
from flask_cors import CORS
import ee
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for frontend access

# ── Earth Engine Initialization ────────────────────────────────────────────────
def initialize_earth_engine():
    """Initialize Google Earth Engine with service account credentials"""
    try:
        # Try to get credentials from environment variables
        private_key = os.environ.get('GEE_PRIVATE_KEY')
        client_email = os.environ.get('GEE_CLIENT_EMAIL')
        project_id = os.environ.get('GEE_PROJECT_ID', 'project-7f7aaf07-19c7-430f-8a3')
        
        if private_key and client_email:
            # Service account authentication
            credentials = ee.ServiceAccountCredentials(client_email, key_data=private_key)
            ee.Initialize(credentials, project=project_id)
            logger.info("Earth Engine initialized with service account")
        else:
            # Try default credentials (for local development)
            ee.Initialize(project=project_id)
            logger.info("Earth Engine initialized with default credentials")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Earth Engine: {e}")
        return False

# Initialize EE on startup
ee_initialized = initialize_earth_engine()

# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_clean_composite(aoi, start_date, end_date, cloud_pct=10):
    """Return a cloud-filtered Sentinel-2 SR median composite clipped to AOI."""
    if not ee_initialized:
        raise Exception("Earth Engine not initialized")
    
    return (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
        .median()
        .clip(aoi)
    )

def compute_ndbi(img):
    """Normalized Difference Built-up Index = (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return img.normalizedDifference(['B11', 'B8']).rename('NDBI')

def compute_ndvi(img):
    """Normalized Difference Vegetation Index = (NIR - Red) / (NIR + Red)."""
    return img.normalizedDifference(['B8', 'B4']).rename('NDVI')

def compute_ndwi(img):
    """Normalized Difference Water Index — used to mask water bodies."""
    return img.normalizedDifference(['B3', 'B8']).rename('NDWI')

def build_construction_score(t1_img, t2_img):
    """
    Construction score = ΔNDBI − ΔNDVI
    High score → NDBI rose (more bare/built) AND NDVI dropped (less green)
    Both signals together strongly indicate active construction.
    """
    delta_ndbi = compute_ndbi(t2_img).subtract(compute_ndbi(t1_img))
    delta_ndvi = compute_ndvi(t2_img).subtract(compute_ndvi(t1_img))
    score = delta_ndbi.subtract(delta_ndvi).rename('change_score')
    
    # Mask water bodies so rivers/ponds don't produce false positives
    water_mask = compute_ndwi(t2_img).lt(0.2)
    return score.updateMask(water_mask)


# ══════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK ENDPOINT
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'ee_initialized': ee_initialized,
        'timestamp': datetime.now().isoformat()
    })


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — True-colour Sentinel-2 tile layer
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/gee-construction', methods=['GET'])
def get_gee_layer():
    """
    Returns a tile URL for a true-colour Sentinel-2 composite of Delhi NCR.
    Query params (all optional):
      date_start  – ISO date, default '2025-10-01'
      date_end    – ISO date, default '2026-03-20'
      cloud_pct   – integer max cloud %, default 10
    """
    try:
        if not ee_initialized:
            return jsonify({'status': 'error', 'message': 'Earth Engine not initialized'}), 503
        
        date_start = request.args.get('date_start', '2025-10-01')
        date_end = request.args.get('date_end', '2026-03-20')
        cloud_pct = int(request.args.get('cloud_pct', 10))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])
        image = get_clean_composite(delhi_ncr, date_start, date_end, cloud_pct)

        vis_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }

        map_info = ee.Image(image).getMapId(vis_params)

        return jsonify({
            'status': 'success',
            'urlFormat': map_info['tile_fetcher'].url_format,
            'dateStart': date_start,
            'dateEnd': date_end
        })

    except Exception as e:
        logger.error(f"Error in get_gee_layer: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 2 — Change-detection score layer
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/gee-change-detection', methods=['GET'])
def get_change_detection():
    """
    Computes ΔNDBI − ΔNDVI between two time windows and returns tile URL.

    Query params (all optional):
      t1_start   – baseline period start,  default '2022-01-01'
      t1_end     – baseline period end,    default '2022-03-31'
      t2_start   – recent period start,    default '2024-01-01'
      t2_end     – recent period end,      default '2024-03-31'
      cloud_pct  – max cloud %, default 10
      threshold  – float, min score shown (0–1), default 0.10
    """
    try:
        if not ee_initialized:
            return jsonify({'status': 'error', 'message': 'Earth Engine not initialized'}), 503
        
        t1_start = request.args.get('t1_start', '2022-01-01')
        t1_end = request.args.get('t1_end', '2022-03-31')
        t2_start = request.args.get('t2_start', '2024-01-01')
        t2_end = request.args.get('t2_end', '2024-03-31')
        cloud_pct = int(request.args.get('cloud_pct', 10))
        threshold = float(request.args.get('threshold', 0.10))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])

        t1_img = get_clean_composite(delhi_ncr, t1_start, t1_end, cloud_pct)
        t2_img = get_clean_composite(delhi_ncr, t2_start, t2_end, cloud_pct)

        score = build_construction_score(t1_img, t2_img)

        # Only show pixels above the threshold
        score_masked = score.updateMask(score.gt(threshold))

        vis_params = {
            'min': threshold,
            'max': 0.5,
            'palette': ['ffffb2', 'fecc5c', 'fd8d3c', 'f03b20', 'bd0026']
        }

        map_info = ee.Image(score_masked).getMapId(vis_params)

        return jsonify({
            'status': 'success',
            'urlFormat': map_info['tile_fetcher'].url_format,
            'baseline': {'start': t1_start, 'end': t1_end},
            'recent': {'start': t2_start, 'end': t2_end},
            'threshold': threshold
        })

    except Exception as e:
        logger.error(f"Error in get_change_detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 3 — Smoothed construction heatmap layer
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/gee-heatmap', methods=['GET'])
def get_heatmap():
    """
    Same change score as endpoint 2 but convolved with a Gaussian kernel
    to produce a smooth density heatmap.

    Query params:
      t1_start, t1_end  – baseline period  (default 2022 Jan–Mar)
      t2_start, t2_end  – recent period    (default 2024 Jan–Mar)
      cloud_pct         – max cloud %,     default 10
      kernel_radius     – Gaussian radius in pixels, default 5
    """
    try:
        if not ee_initialized:
            return jsonify({'status': 'error', 'message': 'Earth Engine not initialized'}), 503
        
        t1_start = request.args.get('t1_start', '2022-01-01')
        t1_end = request.args.get('t1_end', '2022-03-31')
        t2_start = request.args.get('t2_start', '2024-01-01')
        t2_end = request.args.get('t2_end', '2024-03-31')
        cloud_pct = int(request.args.get('cloud_pct', 10))
        kernel_radius = int(request.args.get('kernel_radius', 5))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])

        t1_img = get_clean_composite(delhi_ncr, t1_start, t1_end, cloud_pct)
        t2_img = get_clean_composite(delhi_ncr, t2_start, t2_end, cloud_pct)

        score = build_construction_score(t1_img, t2_img)

        # Only keep positive change (construction, not demolition/greening)
        score_positive = score.max(ee.Image(0))

        # Gaussian smooth → true heatmap feel
        kernel = ee.Kernel.gaussian(
            radius=kernel_radius,
            sigma=kernel_radius / 2,
            units='pixels',
            normalize=True
        )
        smoothed = score_positive.convolve(kernel)

        # Percentile stretch for better visual contrast
        stats = smoothed.reduceRegion(
            reducer=ee.Reducer.percentile([2, 98]),
            geometry=delhi_ncr,
            scale=30,
            maxPixels=1e9
        )
        
        p2 = ee.Number(stats.get('change_score_p2'))
        p98 = ee.Number(stats.get('change_score_p98'))
        norm = smoothed.subtract(p2).divide(p98.subtract(p2)).clamp(0, 1)

        vis_params = {
            'min': 0,
            'max': 1,
            'palette': ['000000', 'ff6b00', 'ff2200', '8b0000']
        }

        map_info = ee.Image(norm).getMapId(vis_params)

        return jsonify({
            'status': 'success',
            'urlFormat': map_info['tile_fetcher'].url_format,
            'baseline': {'start': t1_start, 'end': t1_end},
            'recent': {'start': t2_start, 'end': t2_end},
            'kernelRadius': kernel_radius
        })

    except Exception as e:
        logger.error(f"Error in get_heatmap: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — Quick stats for a bounding box
# ══════════════════════════════════════════════════════════════════════════

@app.route('/api/gee-stats', methods=['GET'])
def get_stats():
    """
    Returns mean change score and estimated changed-area (km²) for the
    map's current bounding box — shown in the dashboard stats strip.

    Query params (required):
      minLng, minLat, maxLng, maxLat  – bounding box from the map
      t1_start, t1_end, t2_start, t2_end  – same as above
    """
    try:
        if not ee_initialized:
            return jsonify({'status': 'error', 'message': 'Earth Engine not initialized'}), 503
        
        min_lng = float(request.args.get('minLng', 76.8))
        min_lat = float(request.args.get('minLat', 28.2))
        max_lng = float(request.args.get('maxLng', 77.6))
        max_lat = float(request.args.get('maxLat', 29.0))
        t1_start = request.args.get('t1_start', '2022-01-01')
        t1_end = request.args.get('t1_end', '2022-03-31')
        t2_start = request.args.get('t2_start', '2024-01-01')
        t2_end = request.args.get('t2_end', '2024-03-31')

        aoi = ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])
        t1_img = get_clean_composite(aoi, t1_start, t1_end)
        t2_img = get_clean_composite(aoi, t2_start, t2_end)
        score = build_construction_score(t1_img, t2_img)

        threshold = 0.10
        changed_px = score.gt(threshold)

        # Get mean score
        mean_stats = score.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        
        # Get changed pixel count
        changed_count = changed_px.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )

        # Get results
        mean_score = mean_stats.getInfo().get('change_score', 0) or 0
        px_count = changed_count.getInfo().get('change_score', 0) or 0
        
        # Calculate area (30m resolution pixels)
        changed_area = round(px_count * (30 * 30 / 1_000_000), 2)

        return jsonify({
            'status': 'success',
            'meanChangeScore': round(float(mean_score), 4),
            'changedAreaKm2': changed_area,
            'baseline': {'start': t1_start, 'end': t1_end},
            'recent': {'start': t2_start, 'end': t2_end}
        })

    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
