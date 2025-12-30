import streamlit as st
import io
import math
import requests
import numpy as np
import cv2
import torch
import concurrent.futures
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time
import bcrypt 
import os
from st_supabase_connection import SupabaseConnection
from streamlit_drawable_canvas import st_canvas

try:
    from transformers import SegformerImageProcessor
except ImportError:
    from transformers import SegformerFeatureExtractor as SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION & DATABASE SETUP
# ==========================================
st.set_page_config(page_title="Zura.ai", layout="wide", page_icon="favicon.ico")

try:
    conn = st.connection(
        "supabase",
        type=SupabaseConnection,
        url=st.secrets["supabase"]["url"],
        key=st.secrets["supabase"]["key"]
    )
except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.stop()

# ==========================================
# 2. AUTHENTICATION LOGIC (SUPABASE)
# ==========================================

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None


def logout():
    st.session_state["authentication_status"] = None
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.rerun()

if st.session_state["authentication_status"] is not True:
    st.title("Welcome to Zura")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username_in = st.text_input("Username", key="login_user")
        password_in = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if username_in and password_in:
                try:
                    response = conn.table("users").select("*").eq("username", username_in).execute()

                    if response.data:
                        user_data = response.data[0]
                        stored_hash = user_data["password"]

                        if check_password(password_in, stored_hash):
                            st.session_state["authentication_status"] = True
                            st.session_state["name"] = user_data["name"]
                            st.session_state["username"] = user_data["username"]
                            st.success(f"Welcome back, {user_data['name']}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
                    else:
                        st.error("User not found.")
                except Exception as e:
                    st.error(f"Login Error: {e}")
            else:
                st.warning("Please enter username and password.")

    with tab2:
        new_name = st.text_input("Full Name")
        new_user = st.text_input("Choose a Username")
        new_pass = st.text_input("Choose a Password", type="password")

        if st.button("Create Account"):
            if new_user and new_pass and new_name:
                try:
                    check = conn.table("users").select("username").eq("username", new_user).execute()
                    if check.data:
                        st.error("Username already exists! Choose another.")
                    else:
                        hashed = hash_password(new_pass)
                        conn.table("users").insert({
                            "username": new_user,
                            "name": new_name,
                            "password": hashed
                        }).execute()
                        st.success("Account created! Please log in.")
                except Exception as e:
                    st.error(f"Signup Error: {e}")
            else:
                st.warning("Please fill all fields.")

# ==========================================
# 3. MAIN APPLICATION (Only if Logged In)
# ==========================================
elif st.session_state["authentication_status"] is True:
    if "models_loaded" not in st.session_state:
        with st.spinner("🚀 Booting up AI Engines (SegFormer + YOLO)..."):
            load_roof_model()
            load_panel_model()
            st.session_state["models_loaded"] = True

    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = ""

    ROOF_MODEL_NAME = "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"
    ZOOM_LEVEL = 19
    IMG_SIZE = 512
    PERFORMANCE_RATIO = 0.75
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def get_lat_lon_from_google(query, api_key):
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        params = {"query": query, "key": api_key}
        try:
            response = requests.get(url, params=params)
            data = response.json()
            if data.get("status") == "OK" and data.get("results"):
                top_result = data["results"][0]
                name = top_result.get("name")
                addr = top_result.get("formatted_address")
                loc = top_result["geometry"]["location"]
                return loc["lat"], loc["lng"], f"{name}, {addr}"
            return None, None, None
        except: return None, None, None

    def fetch_satellite_image(lat, lon, api_key):
        url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {"center": f"{lat},{lon}", "zoom": ZOOM_LEVEL, "size": f"{IMG_SIZE}x{IMG_SIZE}", "maptype": "satellite", "key": api_key}
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            return None
        except: return None

    def fetch_nasa_solar_data(lat, lon):
        try:
            url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
            params = {"parameters": "ALLSKY_SFC_SW_DWN", "community": "RE", "longitude": lon, "latitude": lat, "format": "JSON"}
            resp = requests.get(url, params=params, timeout=5)
            return resp.json()['properties']['parameter']['ALLSKY_SFC_SW_DWN']['ANN']
        except: return 4.5

    @st.cache_resource
    def load_roof_model():
        processor = SegformerImageProcessor.from_pretrained(ROOF_MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(ROOF_MODEL_NAME).to(DEVICE)
        return processor, model

    @st.cache_resource
    def load_panel_model():
        model_filename = "solar_yolo.pt"
        model_url = "https://huggingface.co/finloop/yolov8s-seg-solar-panels/resolve/main/best.pt?download=true"

        if not os.path.exists(model_filename):
            with st.spinner("Downloading Solar Panel AI Model (First Run Only)..."):
                try:
                    response = requests.get(model_url, stream=True)
                    if response.status_code == 200:
                        with open(model_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=1024):
                                f.write(chunk)
                        st.success("Model downloaded successfully!")
                    else:
                        st.warning("Could not download specialized weights. Falling back to standard YOLO.")
                        return YOLO("yolov8n-seg.pt")
                except Exception as e:
                    st.warning(f"Download failed: {e}. Using standard YOLO.")
                    return YOLO("yolov8n-seg.pt")

        try:
            return YOLO(model_filename)
        except:
            return YOLO("yolov8n-seg.pt")

    roof_processor, roof_model = load_roof_model()

    def run_segmentation(image):
        open_cv_image = np.array(image)
        lab = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_image_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        enhanced_image = Image.fromarray(final_image_cv)

        inputs = roof_processor(images=enhanced_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = roof_model(**inputs)

        logits = torch.nn.functional.interpolate(outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        predicted_class_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        raw_mask = (predicted_class_map == 2).astype(np.uint8)
        return raw_mask

    def clean_mask(mask):
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros_like(mask)

        h, w = mask.shape
        center_pt = (w // 2, h // 2)
        best_cnt = None
        max_proximity = -float('inf')

        for cnt in contours:
            if cv2.contourArea(cnt) < 200: continue
            proximity = cv2.pointPolygonTest(cnt, center_pt, True)
            if proximity > max_proximity:
                max_proximity = proximity
                best_cnt = cnt

        final_mask = np.zeros_like(mask)
        if best_cnt is not None:
            epsilon = 0.002 * cv2.arcLength(best_cnt, True)
            approx = cv2.approxPolyDP(best_cnt, epsilon, True)
            cv2.drawContours(final_mask, [approx], -1, 1, thickness=cv2.FILLED)

        return final_mask

    # ==========================================
    # NEW FUNCTIONS: RECENTERING & PANEL DETECTION
    # ==========================================

    def calculate_new_center(lat, lon, mask, zoom=19, img_size=512):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return lat, lon

        largest_cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        if M["m00"] == 0:
            return lat, lon

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        center_x, center_y = img_size // 2, img_size // 2
        dx_px = cX - center_x
        dy_px = cY - center_y 

        meters_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
        shift_x_meters = dx_px * meters_per_px
        shift_y_meters = -dy_px * meters_per_px

        new_lat = lat + (shift_y_meters / 111320.0)
        new_lon = lon + (shift_x_meters / (40075000.0 * math.cos(math.radians(lat)) / 360.0))

        return new_lat, new_lon

    def detect_existing_panels(image, roof_mask):
        model = load_panel_model()
        img_cv = np.array(image)
        results = model.predict(img_cv, conf=0.15, verbose=False)
        final_panel_mask = np.zeros_like(roof_mask)
        total_panel_pixel_area = 0

        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

            for mask in masks:
                mask_resized = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) 

                overlap = cv2.bitwise_and(mask_binary, mask_binary, mask=roof_mask)

                overlap_area = cv2.countNonZero(overlap)
                panel_area = cv2.countNonZero(mask_binary)

                if panel_area > 0 and (overlap_area / panel_area) > 0.3:
                    final_panel_mask = cv2.bitwise_or(final_panel_mask, overlap)
                    total_panel_pixel_area += overlap_area

        return final_panel_mask, total_panel_pixel_area
    # ==========================================
    # 4. SIDEBAR NAVIGATION
    # ==========================================
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Zura AI"


    with st.sidebar:
        st.header("Zura.ai")
        st.markdown("---")

        if st.button("Zura AI", use_container_width=True):
            st.session_state.current_page = "Zura AI"
            st.rerun()

        if st.button("Manual Calculator", use_container_width=True):
            st.session_state.current_page = "Manual Calculator"
            st.rerun()

        if st.button("Visited Sites", use_container_width=True):
            st.session_state.current_page = "Previous Sites"
            st.rerun()

        st.markdown("---")
        st.markdown("<div style='height: 40vh'></div>", unsafe_allow_html=True)
        st.caption(f"Logged in as:")
        st.markdown(f"**{st.session_state['name']}**")

        if st.button("Logout"):
            logout()

    # ==========================================
    # 5. PAGE 1: ROOFTOP ANALYSER (UPDATED)
    # ==========================================
    if st.session_state.current_page == "Zura AI":
        st.title("Zura AI")
        st.markdown("Enter a company name or coordinates to auto-detect the roof.")

        if not api_key:
            st.warning("Google API Key not found in secrets.toml.")

        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Search Location:", placeholder="e.g. Guna Solar Pvt. Ltd., Chennai")

        if "lat" not in st.session_state:
            st.session_state.lat = None
            st.session_state.lon = None
            st.session_state.loc_name = None

        if "calc_usable_area" not in st.session_state:
            st.session_state.calc_usable_area = 0
            st.session_state.calc_total_area = 0
            st.session_state.calc_panel_area = 0
            st.session_state.calc_ghi = 0
            st.session_state.has_run = False
            st.session_state.vis_image = None
            st.session_state.sat_image = None

        if col_btn.button("Find Location") and search_query:
            with st.spinner("Searching..."):
                lat, lon, name_and_addr = get_lat_lon_from_google(search_query, api_key)
                if lat:
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.session_state.loc_name = name_and_addr
                    st.session_state.has_run = False 
                    st.success(f"Found: **{name_and_addr}**")
                else:
                    st.error("Location not found.")

        if st.session_state.lat:
            st.markdown(f"### Analyzing: {st.session_state.loc_name}")

            show_debug = st.checkbox("Show AI Debug View", value=False)

            if st.button("Run Solar Analysis", type="primary"):
                lat = st.session_state.lat
                lon = st.session_state.lon

                with st.status("Starting analysis...", expanded=True) as status:

                    st.write("Refining location center...")
                    initial_img = fetch_satellite_image(lat, lon, api_key)

                    if initial_img:
                        pre_mask = run_segmentation(initial_img)
                        clean_pre_mask = clean_mask(pre_mask) 
                        new_lat, new_lon = calculate_new_center(lat, lon, clean_pre_mask)

                        if new_lat != lat or new_lon != lon:
                            lat = new_lat
                            lon = new_lon
                            st.session_state.lat = lat
                            st.session_state.lon = lon

                    st.write("🛰️ Fetching high-res satellite imagery...")
                    image = fetch_satellite_image(lat, lon, api_key)

                    if image is None:
                        status.update(label="Analysis failed.", state="error")
                        st.error("Failed to load Satellite Image.")
                        st.stop()

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_nasa = executor.submit(fetch_nasa_solar_data, lat, lon)

                    ghi = future_nasa.result()
                    st.write("Solar data fetched.")


                    st.write("Detecting roof structure (SegFormer)...")
                    raw_mask = run_segmentation(image)
                    final_mask = clean_mask(raw_mask)

                    st.write("Scanning for existing panels (YOLOv8 AI)...")
                    panel_mask, panel_pixels = detect_existing_panels(image, final_mask)

                    gsd = 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** ZOOM_LEVEL)
                    pixel_area = gsd ** 2

                    total_roof_area = np.sum(final_mask) * pixel_area
                    existing_panel_area = panel_pixels * pixel_area
                    usable_area = total_roof_area - existing_panel_area

                    vis_img = np.array(image).copy()
                    overlay = np.zeros_like(vis_img)
                    overlay[final_mask == 1] = [0, 255, 0]
                    overlay[panel_mask == 1] = [255, 0, 0]
                    final_vis = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)


                    st.session_state.calc_usable_area = usable_area
                    st.session_state.calc_total_area = total_roof_area
                    st.session_state.calc_panel_area = existing_panel_area
                    st.session_state.calc_ghi = ghi
                    st.session_state.sat_image = image
                    st.session_state.vis_image = final_vis
                    st.session_state.has_run = True

                    status.update(label="Analysis Complete!", state="complete", expanded=False)

            if st.session_state.has_run:
                c1, c2 = st.columns(2)
                with c1: st.image(st.session_state.sat_image, caption="Satellite View", use_container_width=True)
                with c2: st.image(st.session_state.vis_image, caption="Detected Roof (Green) & Panels (Red)", use_container_width=True)

                st.markdown("### Project Feasibility Report")

                st.markdown("#### Select Roof Type")
                roof_type = st.radio(
                    "Select the material of the roof to adjust capacity factor:",
                    ["RCC (Concrete)", "Metal Sheet", "Other / General"],
                    index=2,
                    horizontal=True
                )

                if "RCC" in roof_type:
                    area_factor = 7.0
                elif "Sheet" in roof_type:
                    area_factor = 7.5
                else:
                    area_factor = 8.0

                usable = st.session_state.calc_usable_area
                total = st.session_state.calc_total_area
                panels = st.session_state.calc_panel_area
                ghi = st.session_state.calc_ghi

                capacity_kwp = usable / area_factor
                yearly_gen = capacity_kwp * ghi * PERFORMANCE_RATIO * 365

                m1, m2, m3, m4, m5 = st.columns(5)

                m1.metric(
                    label="Usable Roof Area", 
                    value=f"{usable:,.0f} m²", 
                    delta=f"Total: {total:,.0f} m²", 
                    delta_color="off"
                )
                m2.metric("Existing Panels", f"{panels:,.0f} m²")
                m3.metric("System Capacity", f"{capacity_kwp:,.1f} kWp", help=f"Factor: {area_factor} m²/kW")
                m4.metric("Est. Annual Gen", f"{yearly_gen:,.0f} kWh")
                m5.metric("Irradiance", f"{ghi:.2f} kWh/m²")

                if st.button("Save Project to History"):
                    try:
                        conn.table("searches").insert({
                            "username": st.session_state["username"],
                            "location_name": st.session_state.loc_name,
                            "system_capacity": float(capacity_kwp),
                            "estimated_generation": float(yearly_gen),
                            "roof_area": float(total)
                        }).execute()
                        st.toast("Saved to History!", icon="💾")
                    except Exception as e:
                        st.error(f"Save Error: {e}")


    # ==========================================
    # 6. PAGE 2: PREVIOUS SITES
    # ==========================================
    elif st.session_state.current_page == "Previous Sites":
        st.title("Visited Sites")
        st.markdown(f"Visited sites of **{st.session_state['name']}** from the past 30 days.")

        try:
            rows = conn.table("searches").select("*").eq("username", st.session_state["username"]).order("created_at", desc=True).execute()

            if rows.data:
                df = pd.DataFrame(rows.data)
                df['created_at'] = pd.to_datetime(df['created_at'])

                if df['created_at'].dt.tz is None:
                    df['created_at'] = df['created_at'].dt.tz_localize('UTC')
                df['created_at'] = df['created_at'].dt.tz_convert('Asia/Kolkata')

                st.dataframe(
                    df,
                    column_config={
                        "created_at": st.column_config.DatetimeColumn("Date (IST)", format="D MMM YYYY, h:mm a"),
                        "location_name": "Location",
                        "roof_area": st.column_config.NumberColumn("Area (m²)", format="%.0f"),
                        "system_capacity": st.column_config.NumberColumn("Capacity (kWp)", format="%.1f"),
                        "estimated_generation": st.column_config.NumberColumn("Gen (kWh)", format="%.0f"),
                        "id": None,
                        "username": None
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No projects found. Go to 'Zura AI' to start!")

        except Exception as e:
            st.error(f"Could not load history: {e}")

    # ==========================================
    # 7. PAGE 3: MANUAL CALCULATOR
    # ==========================================
    elif st.session_state.current_page == "Manual Calculator":
        st.title("Manual Area Calculator")
        st.markdown("Search for a location, then click points on the map to trace the roof.")

        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Search Location:", key="manual_search_box")

        if col_btn.button("Find", key="manual_find_btn") and search_query:
            with st.spinner("Loading Map..."):
                lat, lon, name = get_lat_lon_from_google(search_query, api_key)
                if lat:
                    st.session_state.manual_lat = lat
                    st.session_state.manual_lon = lon
                    st.session_state.manual_loc_name = name
                    st.session_state.manual_img = fetch_satellite_image(lat, lon, api_key)
                    st.session_state["canvas_key"] = 0
                else:
                    st.error("Location not found")

        if "manual_img" in st.session_state and st.session_state.manual_img is not None:
            st.write(f"**Location:** {st.session_state.get('manual_loc_name', 'Unknown')}")

            if "canvas_key" not in st.session_state:
                st.session_state["canvas_key"] = 0

            st.info("**INSTRUCTIONS:**\n1. Select the **'Polygon'** tool (star icon).\n2. Click the corners of the roof.\n3. **DOUBLE CLICK** the last point to close the shape.\n4. **RIGHT-CLICK** to exit drawing mode.")

            if st.button("Reset / Clear Drawing"):
                st.session_state["canvas_key"] += 1
                st.rerun()

            canvas_result = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",
                stroke_width=2,
                stroke_color="#00FF00",
                background_image=st.session_state.manual_img,
                update_streamlit=True,
                height=IMG_SIZE,
                width=IMG_SIZE,
                drawing_mode="polygon",
                key=f"manual_canvas_{st.session_state['canvas_key']}",
            )

            if st.button("Calculate Area from Drawing"):
                if canvas_result.json_data is not None and "objects" in canvas_result.json_data:
                    objects = canvas_result.json_data["objects"]

                    if len(objects) > 0:
                        obj = objects[-1]
                        pixel_coords = []


                        if "path" in obj:
                            for point in obj["path"]:
                                if point[0] in ['M', 'L']:
                                    pixel_coords.append([point[1], point[2]])
                        elif "points" in obj:
                            pixel_coords = [[p['x'], p['y']] for p in obj['points']]

                        pixel_coords = np.array(pixel_coords)

                        if len(pixel_coords) > 2:
                            pixel_area = cv2.contourArea(pixel_coords.astype(np.float32))
                            lat = st.session_state.manual_lat
                            gsd = 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** ZOOM_LEVEL)
                            real_area_sqm = pixel_area * (gsd ** 2)

                            manual_factor = 8.0
                            capacity = real_area_sqm / manual_factor
                            generation = capacity * 4.5 * PERFORMANCE_RATIO * 365 

                            st.success("Calculation Successful!")
                            st.markdown("### 📝 Manual Estimation Results")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Drawn Area", f"{real_area_sqm:,.1f} m²")
                            c2.metric("Capacity", f"{capacity:,.1f} kWp")
                            c3.metric("Est. Generation", f"{generation:,.0f} kWh")

                            try:
                                conn.table("searches").insert({
                                    "username": st.session_state["username"],
                                    "location_name": "(Manual) " + st.session_state.manual_loc_name,
                                    "system_capacity": float(capacity),
                                    "estimated_generation": float(generation),
                                    "roof_area": float(real_area_sqm)
                                }).execute()
                                st.toast("Saved to History!", icon="💾")
                            except Exception as e:
                                st.error(f"Save Error: {e}")
                        else:
                            st.warning("⚠️ Shape invalid.")
                    else:
                        st.warning("⚠️ No shape detected.")

    # ==========================================
    # 8. FOOTER
    # ==========================================
    st.markdown("""
    <style>
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #f1f1f1; color: #555; text-align: center;
        padding: 10px; font-size: 12px; border-top: 1px solid #ddd; z-index: 999;
    }
    </style>
    <div class="footer">
        --- AI: <b>SegFormer (Roof) + YOLOv8 (Panels)</b> ---  | ---  Data: <b>NASA POWER</b> ---  | ---  © 2026 Zura.ai ---
    </div>

    """, unsafe_allow_html=True)

