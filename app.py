from flask import Flask, jsonify, request
from flask_cors import CORS
import ee

app = Flask(__name__)
CORS(app)

# ── Initialize Earth Engine ────────────────────────────────────────────────
try:
    print("Initializing Google Earth Engine...")
    ee.Initialize(project='project-7f7aaf07-19c7-430f-8a3')
    print("Earth Engine Initialized Successfully!")
except Exception as e:
    print(f"Failed to initialize Earth Engine: {e}")


# ══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_clean_composite(aoi, start_date, end_date, cloud_pct=10):
    """Return a cloud-filtered Sentinel-2 SR median composite clipped to AOI."""
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
    High score  → NDBI rose (more bare/built) AND NDVI dropped (less green)
    Both signals together strongly indicate active construction.
    """
    delta_ndbi = compute_ndbi(t2_img).subtract(compute_ndbi(t1_img))
    delta_ndvi = compute_ndvi(t2_img).subtract(compute_ndvi(t1_img))
    score      = delta_ndbi.subtract(delta_ndvi).rename('change_score')

    # Mask water bodies so rivers/ponds don't produce false positives
    water_mask = compute_ndwi(t2_img).lt(0.2)
    return score.updateMask(water_mask)


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 1 — True-colour Sentinel-2 tile layer  (your original)
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
        date_start = request.args.get('date_start', '2025-10-01')
        date_end   = request.args.get('date_end',   '2026-03-20')
        cloud_pct  = int(request.args.get('cloud_pct', 10))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])
        image     = get_clean_composite(delhi_ncr, date_start, date_end, cloud_pct)

        vis_params = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }

        map_info = ee.Image(image).getMapId(vis_params)

        return jsonify({
            'status'    : 'success',
            'urlFormat' : map_info['tile_fetcher'].url_format,
            'dateStart' : date_start,
            'dateEnd'   : date_end
        })

    except Exception as e:
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
        t1_start  = request.args.get('t1_start',  '2022-01-01')
        t1_end    = request.args.get('t1_end',    '2022-03-31')
        t2_start  = request.args.get('t2_start',  '2024-01-01')
        t2_end    = request.args.get('t2_end',    '2024-03-31')
        cloud_pct = int(request.args.get('cloud_pct', 10))
        threshold = float(request.args.get('threshold', 0.10))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])

        t1_img = get_clean_composite(delhi_ncr, t1_start, t1_end, cloud_pct)
        t2_img = get_clean_composite(delhi_ncr, t2_start, t2_end, cloud_pct)

        score = build_construction_score(t1_img, t2_img)

        # Only show pixels above the threshold
        score_masked = score.updateMask(score.gt(threshold))

        vis_params = {
            'min'    : threshold,
            'max'    : 0.5,
            'palette': ['ffffb2', 'fecc5c', 'fd8d3c', 'f03b20', 'bd0026']
            #           low change ────────────────────── high change
        }

        map_info = ee.Image(score_masked).getMapId(vis_params)

        return jsonify({
            'status'    : 'success',
            'urlFormat' : map_info['tile_fetcher'].url_format,
            'baseline'  : {'start': t1_start, 'end': t1_end},
            'recent'    : {'start': t2_start, 'end': t2_end},
            'threshold' : threshold
        })

    except Exception as e:
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
      opacity           – float 0–1,       default 0.75
    """
    try:
        t1_start      = request.args.get('t1_start',      '2022-01-01')
        t1_end        = request.args.get('t1_end',        '2022-03-31')
        t2_start      = request.args.get('t2_start',      '2024-01-01')
        t2_end        = request.args.get('t2_end',        '2024-03-31')
        cloud_pct     = int(request.args.get('cloud_pct',     10))
        kernel_radius = int(request.args.get('kernel_radius',  5))

        delhi_ncr = ee.Geometry.Rectangle([76.8, 28.2, 77.6, 29.0])

        t1_img = get_clean_composite(delhi_ncr, t1_start, t1_end, cloud_pct)
        t2_img = get_clean_composite(delhi_ncr, t2_start, t2_end, cloud_pct)

        score = build_construction_score(t1_img, t2_img)

        # Only keep positive change (construction, not demolition/greening)
        score_positive = score.max(ee.Image(0))

        # Gaussian smooth → true heatmap feel
        kernel    = ee.Kernel.gaussian(
            radius=kernel_radius,
            sigma=kernel_radius / 2,
            units='pixels',
            normalize=True
        )
        smoothed  = score_positive.convolve(kernel)

        # Percentile stretch for better visual contrast
        stats     = smoothed.reduceRegion(
            reducer   = ee.Reducer.percentile([2, 98]),
            geometry  = delhi_ncr,
            scale     = 30,
            maxPixels = 1e9
        )
        p2   = ee.Number(stats.get('change_score_p2'))
        p98  = ee.Number(stats.get('change_score_p98'))
        norm = smoothed.subtract(p2).divide(p98.subtract(p2)).clamp(0, 1)

        vis_params = {
            'min'    : 0,
            'max'    : 1,
            # Transparent black → orange → red  (alpha via opacity in frontend)
            'palette': ['000000', 'ff6b00', 'ff2200', '8b0000']
        }

        map_info = ee.Image(norm).getMapId(vis_params)

        return jsonify({
            'status'      : 'success',
            'urlFormat'   : map_info['tile_fetcher'].url_format,
            'baseline'    : {'start': t1_start, 'end': t1_end},
            'recent'      : {'start': t2_start, 'end': t2_end},
            'kernelRadius': kernel_radius
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════
#  ENDPOINT 4 — Quick stats for a bounding box (used by the dashboard)
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
        min_lng   = float(request.args.get('minLng', 76.8))
        min_lat   = float(request.args.get('minLat', 28.2))
        max_lng   = float(request.args.get('maxLng', 77.6))
        max_lat   = float(request.args.get('maxLat', 29.0))
        t1_start  = request.args.get('t1_start', '2022-01-01')
        t1_end    = request.args.get('t1_end',   '2022-03-31')
        t2_start  = request.args.get('t2_start', '2024-01-01')
        t2_end    = request.args.get('t2_end',   '2024-03-31')

        aoi    = ee.Geometry.Rectangle([min_lng, min_lat, max_lng, max_lat])
        t1_img = get_clean_composite(aoi, t1_start, t1_end)
        t2_img = get_clean_composite(aoi, t2_start, t2_end)
        score  = build_construction_score(t1_img, t2_img)

        threshold     = 0.10
        changed_px    = score.gt(threshold)

        # Count pixels above threshold → multiply by pixel area (100 m² at 10 m)
        pixel_area_km2 = 10 * 10 / 1_000_000   # = 0.0001 km²

        stats = score.reduceRegion(
            reducer   = ee.Reducer.mean().combine(
                            ee.Reducer.count(), sharedInputs=True),
            geometry  = aoi,
            scale     = 30,
            maxPixels = 1e9
        )
        changed_count = changed_px.reduceRegion(
            reducer   = ee.Reducer.sum(),
            geometry  = aoi,
            scale     = 30,
            maxPixels = 1e9
        )

        mean_score    = stats.getInfo().get('change_score_mean', 0) or 0
        px_count      = changed_count.getInfo().get('change_score', 0) or 0
        changed_area  = round(px_count * (30 * 30 / 1_000_000), 2)   # 30 m pixels

        return jsonify({
            'status'          : 'success',
            'meanChangeScore' : round(float(mean_score), 4),
            'changedAreaKm2'  : changed_area,
            'baseline'        : {'start': t1_start, 'end': t1_end},
            'recent'          : {'start': t2_start, 'end': t2_end}
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
