# Zura.ai - AI-Powered Solar Rooftop Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A production-ready web application that leverages computer vision and deep learning to automatically detect rooftop structures, identify existing solar panel installations, and estimate solar energy generation potential from satellite imagery.

---

## Overview

Zura.ai combines semantic segmentation (SegFormer) and object detection (YOLOv8) models to provide automated solar feasibility assessments. The platform integrates Google Maps API for geospatial data retrieval and NASA POWER API for solar irradiance metrics, delivering actionable insights for solar installation planning.

### Key Capabilities

- **Automated Roof Detection**: Semantic segmentation using SegFormer transformer architecture
- **Panel Recognition**: YOLOv8-based instance segmentation for existing solar installations
- **Geographic Recentering**: Automatic correction of map coordinates to building centroids
- **Solar Analytics**: Energy generation estimates based on NASA climatological data
- **Manual Calculation Mode**: Interactive polygon drawing for custom area measurements
- **User Management**: Secure authentication with bcrypt password hashing
- **Project History**: Persistent storage of analysis results via Supabase

---

## Architecture

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit (Python web framework) |
| **Authentication** | bcrypt + Supabase |
| **Database** | Supabase (PostgreSQL) |
| **Roof Segmentation** | SegFormer-B2 (wu-pr-gw/segformer-b2-finetuned-with-LoveDA) |
| **Panel Detection** | YOLOv8s-seg (finloop/yolov8s-seg-solar-panels) |
| **Image Processing** | OpenCV, PIL |
| **Geospatial APIs** | Google Maps Static/Places API |
| **Solar Data** | NASA POWER API |

### Model Pipeline

```
Input (Location Query)
    ↓
Google Places API → Geocoding (Lat/Lon)
    ↓
Google Static Maps API → Satellite Image (512×512, Zoom 19)
    ↓
SegFormer Inference → Building Mask
    ↓
Centroid Calculation → Recentered Coordinates
    ↓
Refined Satellite Image Fetch
    ↓
YOLOv8 Inference → Existing Panel Masks
    ↓
Geometric Calculations → Area Metrics
    ↓
NASA POWER API → GHI (Global Horizontal Irradiance)
    ↓
Energy Estimation (kWh/year)
```

---

## Installation

### Prerequisites

```bash
Python >= 3.8
CUDA-compatible GPU (optional, for faster inference)
```

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/zura-ai.git
cd zura-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
streamlit-drawable-canvas>=0.9.3
st-supabase-connection>=0.2.0
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
bcrypt>=4.0.0
requests>=2.31.0
matplotlib>=3.7.0
```

### 3. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key"

GOOGLE_API_KEY = "your-google-maps-api-key"
```

### 4. Database Setup

Execute in Supabase SQL Editor:

```sql
-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Searches table
CREATE TABLE searches (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) REFERENCES users(username),
    location_name TEXT,
    system_capacity FLOAT,
    estimated_generation FLOAT,
    roof_area FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_searches_username ON searches(username);
CREATE INDEX idx_searches_created_at ON searches(created_at DESC);
```

---

## Usage

### Launch Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### Workflow

#### 1. Authentication
- **Sign Up**: Create account with username/password (bcrypt-hashed)
- **Login**: Authenticate to access platform

#### 2. AI Analysis Mode
1. Enter location query (company name, address, or coordinates)
2. Click **Find Location** → System geocodes and fetches satellite imagery
3. Click **Run Solar Analysis** → Executes AI pipeline:
   - SegFormer detects building footprint
   - YOLOv8 identifies existing panels
   - NASA API retrieves solar irradiance data
4. Select **Roof Type** (RCC/Metal/Other) for capacity factor adjustment
5. Review metrics: Usable area, capacity (kWp), annual generation (kWh)
6. Save project to history

#### 3. Manual Calculator Mode
1. Search location
2. Use polygon tool to manually trace rooftop
3. Double-click to close shape
4. Calculate area with real-world coordinate transformation

#### 4. Project History
- View all saved analyses with timestamps
- Export-ready tabular format

---

## Technical Implementation

### Computer Vision Pipeline

#### Roof Detection (SegFormer)

```python
# Preprocessing: CLAHE enhancement in LAB color space
lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
enhanced = cv2.merge((clahe.apply(l), a, b))

# Inference
inputs = processor(images=enhanced, return_tensors="pt").to(DEVICE)
outputs = model(**inputs)
mask = outputs.logits.argmax(dim=1)[0]  # Class 2 = Buildings
```

#### Mask Refinement

```python
# Morphological operations
closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
dilated = cv2.dilate(closed, kernel, iterations=1)

# Select largest center-proximal contour
contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best_contour = max(contours, key=lambda c: cv2.pointPolygonTest(c, center, True))
```

#### Panel Detection (YOLOv8)

```python
results = yolo_model.predict(image, conf=0.15)

# Strict roof constraint: Only accept panels within detected building mask
for panel_mask in results[0].masks:
    overlap = cv2.bitwise_and(panel_mask, roof_mask)
    if (overlap_area / panel_area) > 0.3:  # 30% threshold
        final_panels = cv2.bitwise_or(final_panels, overlap)
```

### Geospatial Calculations

#### Coordinate Recentering

```python
# Ground Sample Distance (meters/pixel)
gsd = 156543.03392 * cos(lat_rad) / (2 ** zoom_level)

# Pixel shift from image center to building centroid
dx_meters = (centroid_x - img_center_x) * gsd
dy_meters = -(centroid_y - img_center_y) * gsd

# Degrees conversion
new_lat = lat + (dy_meters / 111320.0)
new_lon = lon + (dx_meters / (40075000 * cos(lat_rad) / 360))
```

#### Area Calculation

```python
pixel_area_m2 = gsd ** 2
total_roof_area = np.sum(roof_mask) * pixel_area_m2
panel_area = np.sum(panel_mask) * pixel_area_m2
usable_area = total_roof_area - panel_area
```

### Energy Estimation

```python
# Capacity factors (m²/kWp) by roof type
capacity_factors = {
    "RCC": 7.0,
    "Metal": 7.5,
    "Other": 8.0
}

# System capacity
capacity_kwp = usable_area / capacity_factor

# Annual generation
performance_ratio = 0.75
ghi = nasa_irradiance_kwh_per_m2_per_day
annual_kwh = capacity_kwp * ghi * performance_ratio * 365
```

---

## API Configuration

### Google Maps APIs

**Required APIs:**
- Maps Static API
- Places API (Text Search)

**Billing:** Ensure billing enabled on Google Cloud Console

**Sample Request:**
```python
# Geocoding
url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
params = {"query": "Company Name, City", "key": API_KEY}

# Satellite Image
url = "https://maps.googleapis.com/maps/api/staticmap"
params = {
    "center": "lat,lon",
    "zoom": 19,
    "size": "512x512",
    "maptype": "satellite",
    "key": API_KEY
}
```

### NASA POWER API

**Endpoint:** https://power.larc.nasa.gov/api/temporal/climatology/point

**Parameter:** `ALLSKY_SFC_SW_DWN` (All Sky Surface Shortwave Downward Irradiance)

**No authentication required**

```python
params = {
    "parameters": "ALLSKY_SFC_SW_DWN",
    "community": "RE",  # Renewable Energy
    "longitude": lon,
    "latitude": lat,
    "format": "JSON"
}
```

---

## Model Details

### SegFormer (Roof Segmentation)

- **Architecture**: SegFormer-B2 (Hierarchical Transformer)
- **Training Dataset**: LoveDA (Land-cover Dataset)
- **Classes**: Background, Other, Buildings (Class 2 used)
- **Input Size**: 512×512 RGB
- **Output**: Semantic segmentation mask
- **Inference Device**: CUDA/MPS/CPU (auto-detected)

**Model Card**: [wu-pr-gw/segformer-b2-finetuned-with-LoveDA](https://huggingface.co/wu-pr-gw/segformer-b2-finetuned-with-LoveDA)

### YOLOv8 (Panel Detection)

- **Architecture**: YOLOv8s-seg (Segmentation variant)
- **Training Dataset**: Solar panel aerial imagery
- **Confidence Threshold**: 0.15
- **Post-processing**: Roof mask constraint (30% overlap minimum)
- **Auto-download**: Weights fetched from HuggingFace on first run

**Model Card**: [finloop/yolov8s-seg-solar-panels](https://huggingface.co/finloop/yolov8s-seg-solar-panels)

---

## Performance Optimization

### Caching Strategy

```python
@st.cache_resource
def load_roof_model():
    # Models loaded once per session
    return processor, model
```

### Concurrent API Calls

```python
with concurrent.futures.ThreadPoolExecutor() as executor:
    future_nasa = executor.submit(fetch_nasa_solar_data, lat, lon)
    ghi = future_nasa.result()
```

### Device Selection

```python
DEVICE = "cuda" if torch.cuda.is_available() else          ("mps" if torch.backends.mps.is_available() else "cpu")
```

---

## Security

### Password Hashing

```python
import bcrypt

# Registration
hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Login verification
bcrypt.checkpw(input_password.encode('utf-8'), stored_hash.encode('utf-8'))
```

### Session Management

```python
st.session_state["authentication_status"] = True
st.session_state["username"] = username
```

### Environment Variables

All sensitive credentials stored in `.streamlit/secrets.toml` (gitignored)

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ZOOM_LEVEL` | 19 | Google Maps zoom (higher = more detail) |
| `IMG_SIZE` | 512 | Image dimensions (pixels) |
| `PERFORMANCE_RATIO` | 0.75 | System efficiency factor |
| `CAPACITY_FACTOR_RCC` | 7.0 | Area per kWp for concrete roofs |
| `CAPACITY_FACTOR_METAL` | 7.5 | Area per kWp for metal roofs |
| `CAPACITY_FACTOR_OTHER` | 8.0 | Area per kWp for general roofs |
| `PANEL_OVERLAP_THRESHOLD` | 0.3 | Min overlap with roof to validate panel |
| `MIN_CONTOUR_AREA` | 200 | Min pixels to consider building contour |

---

## Limitations

1. **Geographic Coverage**: Google Static Maps API pricing may limit large-scale deployments
2. **Resolution**: 512×512 images at zoom 19 (~0.3m/pixel at equator)
3. **Roof Types**: Detection optimized for flat/low-slope roofs
4. **Shadowing**: No shade analysis (trees, adjacent buildings)
5. **Panel Orientation**: Assumes optimal tilt angles
6. **Inverter Losses**: Not modeled separately (included in performance ratio)

---

## Roadmap

- [ ] Time-series solar generation forecasting
- [ ] 3D roof modeling from LiDAR data
- [ ] Shade analysis using building height maps
- [ ] Financial ROI calculator with utility rate integration
- [ ] Multi-language support
- [ ] Mobile-responsive UI
- [ ] Batch processing for commercial portfolios
- [ ] REST API for third-party integrations

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Open a Pull Request

**Development Setup:**
```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Citation

If you use this platform in research, please cite:

```bibtex
@software{zura_ai_2025,
  author = {Your Name},
  title = {Zura.ai: AI-Powered Solar Rooftop Analysis Platform},
  year = {2025},
  url = {https://github.com/yourusername/zura-ai}
}
```

---

## Acknowledgments

- **SegFormer Model**: [wu-pr-gw](https://huggingface.co/wu-pr-gw)
- **YOLOv8 Solar Weights**: [finloop](https://huggingface.co/finloop)
- **Solar Data**: NASA POWER Project
- **Geospatial APIs**: Google Cloud Platform

---

## Support

For issues, questions, or feature requests:
- **GitHub Issues**: [Submit Issue](https://github.com/yourusername/zura-ai/issues)
- **Email**: your.email@example.com

---

**Built with ❤️ for the solar energy transition**
