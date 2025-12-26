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
import bcrypt  # <--- NEW: For secure password hashing

# --- SUPABASE CONNECTION ---
from st_supabase_connection import SupabaseConnection

# --- MANUAL DRAWING ---
from streamlit_drawable_canvas import st_canvas

# --- AI TRANSFORMERS ---
try:
    from transformers import SegformerImageProcessor
except ImportError:
    from transformers import SegformerFeatureExtractor as SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation


# ==========================================
# 1. CONFIGURATION & DATABASE SETUP
# ==========================================
st.set_page_config(page_title="Zura.ai", layout="wide", page_icon="favicon.ico")

# Initialize Supabase
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

# Helper: Hash Passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Helper: Check Passwords
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Initialize Session State for Auth
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None
if "name" not in st.session_state:
    st.session_state["name"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None

# Logout Function
def logout():
    st.session_state["authentication_status"] = None
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.rerun()

# --- LOGIN / SIGNUP UI ---
if st.session_state["authentication_status"] is not True:
    st.title("☀️ Welcome to Zura")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # LOGIN TAB
    with tab1:
        username_in = st.text_input("Username", key="login_user")
        password_in = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login"):
            if username_in and password_in:
                try:
                    # Fetch user from Supabase
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

    # SIGNUP TAB
    with tab2:
        new_name = st.text_input("Full Name")
        new_user = st.text_input("Choose a Username")
        new_pass = st.text_input("Choose a Password", type="password")
        
        if st.button("Create Account"):
            if new_user and new_pass and new_name:
                try:
                    # Check if user exists
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
    
    # --- GLOBAL VARIABLES ---
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = ""

    MODEL_NAME = "wu-pr-gw/segformer-b2-finetuned-with-LoveDA"
    ZOOM_LEVEL = 19
    IMG_SIZE = 512
    AREA_TO_KW_FACTOR = 8.0  
    PERFORMANCE_RATIO = 0.75
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # --- HELPER FUNCTIONS ---
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
    def load_ai_model():
        processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
        return processor, model

    processor, model = load_ai_model()

    def run_segmentation(image):
        # Preprocessing
        open_cv_image = np.array(image)
        lab = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_image_cv = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        enhanced_image = Image.fromarray(final_image_cv)

        # Inference
        inputs = processor(images=enhanced_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = torch.nn.functional.interpolate(outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False)
        predicted_class_map = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        
        # Extract only Class 2 (Buildings)
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
    # 5. PAGE 1: ROOFTOP ANALYSER
    # ==========================================
    if st.session_state.current_page == "Zura AI":
        st.title("Zura AI")
        st.markdown("Enter a company name or coordinates to auto-detect the roof.")

        if not api_key:
            st.warning("⚠️ Google API Key not found in secrets.toml.")

        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Search Location:", placeholder="e.g. Guna Solar Pvt. Ltd., Chennai")

        if "lat" not in st.session_state:
            st.session_state.lat = None
            st.session_state.lon = None
            st.session_state.loc_name = None

        if col_btn.button("Find Location") and search_query:
            with st.spinner("Searching..."):
                lat, lon, name_and_addr = get_lat_lon_from_google(search_query, api_key)
                if lat:
                    st.session_state.lat = lat
                    st.session_state.lon = lon
                    st.session_state.loc_name = name_and_addr
                    st.success(f"📍 Found: **{name_and_addr}**")
                else:
                    st.error("❌ Location not found.")

        if st.session_state.lat:
            st.markdown(f"### Analyzing: {st.session_state.loc_name}")
            
            show_debug = st.checkbox("🛠️ Show AI Debug View (Check this if roof is not detected)", value=False)
            
            if st.button("⚡ Run Solar Analysis", type="primary"):
                lat = st.session_state.lat
                lon = st.session_state.lon
                
                with st.status("Starting analysis...", expanded=True) as status:
                    st.write("🛰️ Fetching satellite imagery...")
                    image = fetch_satellite_image(lat, lon, api_key)
                    
                    if image is None:
                        status.update(label="Analysis failed.", state="error")
                        st.error("⚠️ Failed to load Satellite Image. Check your API Key.")
                        st.stop()
                    
                    # --- DEBUG MODE ---
                    if show_debug:
                        st.write("🔍 Running Debug Segmentation...")
                        raw_debug = run_segmentation(image)
                        st.image(raw_debug * 255, caption="AI Raw Mask", use_column_width=True)
                        status.update(label="Debug Complete", state="complete")
                        st.stop()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future_nasa = executor.submit(fetch_nasa_solar_data, lat, lon)
                    
                    ghi = future_nasa.result()
                    st.write("✅ Data fetched successfully.")

                    st.write("🧠 Running AI model to detect roof...")
                    raw_mask = run_segmentation(image)
                    st.write("✅ AI detection complete.")

                    st.write("📐 Isolating central building and refining geometry...")
                    final_mask = clean_mask(raw_mask)
                    
                    # Maths
                    gsd = 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** ZOOM_LEVEL)
                    pixel_area = gsd ** 2
                    roof_area_sqm = np.sum(final_mask) * pixel_area
                    capacity_kwp = roof_area_sqm / AREA_TO_KW_FACTOR
                    yearly_gen = capacity_kwp * ghi * PERFORMANCE_RATIO * 365
                    
                    # Visualization
                    vis_img = np.array(image).copy()
                    overlay = np.zeros_like(vis_img)
                    overlay[final_mask == 1] = [0, 255, 0]
                    final_vis = cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0)

                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                # Display Results
                c1, c2 = st.columns(2)
                with c1: st.image(image, caption="Satellite View", use_column_width=True)
                with c2: st.image(final_vis, caption="AI Detected Roof (Isolated)", use_column_width=True)
                
                st.markdown("### 📊 Project Feasibility Report")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Roof Area", f"{roof_area_sqm:,.0f} m²")
                m2.metric("System Capacity", f"{capacity_kwp:,.1f} kWp")
                m3.metric("Annual Gen", f"{yearly_gen:,.0f} kWh")
                m4.metric("Irradiance", f"{ghi:.2f} kWh/m²")
                
                # Save to DB
                try:
                    conn.table("searches").insert({
                        "username": st.session_state["username"],
                        "location_name": st.session_state.loc_name,
                        "system_capacity": float(capacity_kwp),
                        "estimated_generation": float(yearly_gen),
                        "roof_area": float(roof_area_sqm)
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
                
                # Convert Timezone (UTC -> IST)
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

            st.info("👇 **INSTRUCTIONS:**\n1. Select the **'Polygon'** tool (star icon).\n2. Click the corners of the roof.\n3. **DOUBLE CLICK** the last point to close the shape.\n4. **RIGHT-CLICK** to exit drawing mode.")

            if st.button("♻️ Reset / Clear Drawing"):
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

            if st.button("🧮 Calculate Area from Drawing"):
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

                            capacity = real_area_sqm / AREA_TO_KW_FACTOR
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
        --- AI: <b>SegFormer</b> ---  | ---  Data: <b>NASA POWER</b> ---  | ---  Manual: <b>Canvas</b> ---  | ---  © 2026 Zura.ai ---
    </div>

    """, unsafe_allow_html=True)
