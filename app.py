import os
import math
import base64
import requests
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

db = SQLAlchemy()

def create_app():
    # ==========================================
    # DATABASE CONFIGURATION (Railway Ready)
    # ==========================================
    db_url = os.environ.get('DATABASE_URL')
    
    # Railway sometimes provides 'postgres://', SQLAlchemy needs 'postgresql://'
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    # Fallback to local SQLite
    if not db_url:
        # Use /tmp for ephemeral storage if we can't write to the current directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, 'pool_history.db')
        db_url = f"sqlite:///{db_path}"
        print(f"DEBUG: Using SQLite at {db_path}")

    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        try:
            db.create_all()
            print("DATABASE: Tables verified/created successfully.")
        except Exception as e:
            print(f"DATABASE ERROR: {e}")
            # Non-fatal during startup to allow the app to at least boot
    
    return app

class Scan(db.Model):
    __tablename__ = 'scans'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String, nullable=False, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fcl = db.Column(db.Float)
    alk = db.Column(db.Float)
    ph = db.Column(db.Float)
    th = db.Column(db.Float)
    weather_trend = db.Column(db.String)

# Initialize the app and DB
create_app()

# ==========================================
# 1. THE VISION MATRIX
# ==========================================
brands = {
    "aquarius_4way": {
        "fcl": [{"rgb": (250, 250, 240), "val": 0.0}, {"rgb": (230, 200, 220), "val": 1.0}, {"rgb": (212, 169, 201), "val": 3.0}, {"rgb": (180, 100, 180), "val": 5.0}, {"rgb": (140, 70, 160), "val": 10.0}],
        "alk": [{"rgb": (220, 180, 40), "val": 0}, {"rgb": (180, 170, 40), "val": 40}, {"rgb": (154, 158, 72), "val": 80}, {"rgb": (100, 120, 50), "val": 120}, {"rgb": (50, 80, 40), "val": 240}],
        "ph": [{"rgb": (230, 160, 40), "val": 6.8}, {"rgb": (222, 138, 50), "val": 7.4}, {"rgb": (200, 80, 40), "val": 7.8}, {"rgb": (180, 40, 40), "val": 8.4}],
        "th": [{"rgb": (150, 180, 200), "val": 0}, {"rgb": (80, 130, 170), "val": 100}, {"rgb": (67, 113, 122), "val": 250}, {"rgb": (80, 60, 120), "val": 500}]
    },
    "clorox_6way": {
        "fcl": [{"rgb": (250, 250, 240), "val": 0.0}, {"rgb": (212, 169, 201), "val": 3.0}, {"rgb": (140, 70, 160), "val": 10.0}],
        "alk": [{"rgb": (220, 180, 40), "val": 0}, {"rgb": (154, 158, 72), "val": 80}, {"rgb": (50, 80, 40), "val": 240}],
        "ph": [{"rgb": (230, 160, 40), "val": 6.8}, {"rgb": (222, 138, 50), "val": 7.4}, {"rgb": (180, 40, 40), "val": 8.4}],
        "th": [{"rgb": (150, 180, 200), "val": 0}, {"rgb": (67, 113, 122), "val": 250}, {"rgb": (80, 60, 120), "val": 500}]
    },
    "aquachek_7way": {
        "fcl": [{"rgb": (250, 250, 240), "val": 0.0}, {"rgb": (212, 169, 201), "val": 3.0}, {"rgb": (140, 70, 160), "val": 10.0}],
        "alk": [{"rgb": (220, 180, 40), "val": 0}, {"rgb": (154, 158, 72), "val": 80}, {"rgb": (50, 80, 40), "val": 240}],
        "ph": [{"rgb": (230, 160, 40), "val": 6.8}, {"rgb": (222, 138, 50), "val": 7.4}, {"rgb": (180, 40, 40), "val": 8.4}],
        "th": [{"rgb": (150, 180, 200), "val": 0}, {"rgb": (67, 113, 122), "val": 250}, {"rgb": (80, 60, 120), "val": 500}]
    },
    "hth_6way": {
        "fcl": [{"rgb": (250, 250, 240), "val": 0.0}, {"rgb": (212, 169, 201), "val": 3.0}, {"rgb": (140, 70, 160), "val": 10.0}],
        "alk": [{"rgb": (220, 180, 40), "val": 0}, {"rgb": (154, 158, 72), "val": 80}, {"rgb": (50, 80, 40), "val": 240}],
        "ph": [{"rgb": (230, 160, 40), "val": 6.8}, {"rgb": (222, 138, 50), "val": 7.4}, {"rgb": (180, 40, 40), "val": 8.4}],
        "th": [{"rgb": (150, 180, 200), "val": 0}, {"rgb": (67, 113, 122), "val": 250}, {"rgb": (80, 60, 120), "val": 500}]
    }
}

def find_closest_match(target_rgb, scale_list):
    closest_val = None
    min_distance = float('inf')
    for item in scale_list:
        distance = math.sqrt((target_rgb[0] - item["rgb"][0])**2 + (target_rgb[1] - item["rgb"][1])**2 + (target_rgb[2] - item["rgb"][2])**2)
        if distance < min_distance:
            min_distance = distance
            closest_val = item["val"]
    return closest_val

# ==========================================
# 2. THE CORE ENGINE (Math & Memory)
# ==========================================
def process_pool_data(fcl_val, alk_val, ph_val, th_val=250.0, volume_l=20000):
    # Live Weather (Quinte West)
    target_fcl = 3.0
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=44.18&longitude=-77.57&current=temperature_2m,precipitation,uv_index"
        res = requests.get(url, timeout=5).json()
        temp = res['current']['temperature_2m']
        precip = res['current']['precipitation']
        uv = res['current']['uv_index']
        
        weather_desc = 'Raining' if precip > 0 else 'Dry'
        weather_trend = f"{temp}°C, UV: {uv}, {weather_desc}"
        
        if temp >= 28.0 or uv >= 6.0:
            target_fcl = 4.0
    except Exception as e:
        weather_trend = "Weather unavailable"
        app.logger.warning(f"Weather API error: {e}")

    # Save to Database (SQLAlchemy handles sessions safely)
    try:
        new_scan = Scan(
            fcl=fcl_val,
            alk=alk_val,
            ph=ph_val,
            th=th_val,
            weather_trend=weather_trend
        )
        db.session.add(new_scan)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Database error: {e}")

    # The Math Engine (Dynamic Volume)
    plan = []
    vol_multiplier = volume_l / 20000.0

    if ph_val > 7.6:
        dosage = round(((ph_val - 7.4) / 0.2) * 200 * vol_multiplier)
        plan.append({"title": f"Lower pH ({ph_val})", "action": f"Add {dosage}g of pH Down.", "color": "red"})
    elif ph_val < 7.2:
        dosage = round(((7.4 - ph_val) / 0.2) * 100 * vol_multiplier)
        plan.append({"title": f"Raise pH ({ph_val})", "action": f"Add {dosage}g of pH Up.", "color": "orange"})

    fcl_threshold = target_fcl - 1.0
    if fcl_val < fcl_threshold:
        dosage = round((target_fcl - fcl_val) * 60 * vol_multiplier)
        plan.append({"title": f"Low Chlorine ({fcl_val})", "action": f"Add {dosage}g of Turbo Shock.", "color": "yellow"})

    if not plan:
        plan.append({"title": "Swim Ready!", "action": "Levels are golden. Say hi to the rubber ducky floatie.", "color": "green"})

    return plan

# ==========================================
# 3. THE ROUTES
# ==========================================
@app.route('/')
def home(): return render_template('index.html')

@app.route('/history')
def view_history():
    try:
        scans = Scan.query.order_by(Scan.timestamp.desc()).limit(10).all()
    except Exception as e:
        app.logger.error(f"History fetch error: {e}")
        scans = []
    
    chart_scans = list(reversed(scans))
    
    labels = [scan.timestamp.split()[0][-5:] for scan in chart_scans] if chart_scans else []
    fcl_data = [scan.fcl for scan in chart_scans] if chart_scans else []
    ph_data = [scan.ph for scan in chart_scans] if chart_scans else []
    
    # Maintain tuple format for backwards compatibility with existing templates
    scan_tuples = [(s.id, s.timestamp, s.fcl, s.alk, s.ph, s.th, s.weather_trend) for s in scans]
    
    return render_template('history.html', scans=scan_tuples, labels=labels, fcl_data=fcl_data, ph_data=ph_data)

@app.route('/scan', methods=['POST'])
def scan_manual():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    volume = float(data.get("volume", 20000))
    plan = process_pool_data(
        float(data.get("fcl", 1.0)), 
        float(data.get("alk", 80)), 
        float(data.get("ph", 8.2)), 
        float(data.get("th", 250)),
        volume_l=volume
    )
    return jsonify({"treatment_plan": plan})

@app.route('/analyze_pixels', methods=['POST'])
def analyze_pixels():
    data = request.json
    b64_data = data.get('image_base64')
    coords = data.get('coords') # [{"x": 100, "y": 200}, ...]
    brand = data.get('brand', 'aquarius_4way')
    volume = float(data.get('volume', 20000))
    
    if not b64_data or len(coords) != 4:
        return jsonify({"error": "Invalid payload"}), 400

    scale_dict = brands.get(brand)
    if not scale_dict:
        return jsonify({"error": "Unsupported brand selected."}), 400

    try:
        header, encoded = b64_data.split(",", 1)
        file_bytes = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        app.logger.error(f"Image processing error: {e}")
        return jsonify({"error": f"Image processing error: {str(e)}"}), 400
    
    results = {}
    pad_names = ["fcl", "alk", "ph", "th"]
    
    for i, pad_name in enumerate(pad_names):
        x = coords[i]['x']
        y = coords[i]['y']
        
        y_start = max(0, y - 5)
        y_end = min(img_rgb.shape[0], y + 5)
        x_start = max(0, x - 5)
        x_end = min(img_rgb.shape[1], x + 5)
        
        pad_crop = img_rgb[y_start:y_end, x_start:x_end]
        
        if pad_crop.size == 0:
            final_rgb = (255, 255, 255)
        else:
            avg_color = np.average(np.average(pad_crop, axis=0), axis=0)
            final_rgb = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
            
        results[pad_name] = find_closest_match(final_rgb, scale_dict[pad_name])

    plan = process_pool_data(
        results["fcl"], 
        results["alk"], 
        results["ph"], 
        results.get("th", 250),
        volume_l=volume
    )
    
    return jsonify({
        "detected_levels": results,
        "treatment_plan": plan
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)