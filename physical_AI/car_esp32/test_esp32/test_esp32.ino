#include <WiFi.h>
#include <WebServer.h>

// ============================================================
//  WIFI SETTINGS
// ============================================================
const char* WIFI_SSID = "laptop";
const char* WIFI_PASS = "12345678";

WebServer server(80);

// ============================================================
//  HARDWARE PINS
// ============================================================
#define IN1 26
#define IN2 27
#define IN3 14
#define IN4 13
#define ENA 32
#define ENB 33

// Start with a very slow test speed
int speedPWM = 90; 

// ============================================================
//  HTML DASHBOARD (Mobile & Desktop Friendly)
// ============================================================
const char* html_page = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Diagnostic Test</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background-color: #1e1e1e; color: #fff; margin-top: 20px; }
    h2 { color: #4ade80; margin-bottom: 5px; }
    p { color: #aaa; font-size: 14px; margin-top: 0; }
    .btn { 
      display: inline-block; width: 110px; height: 60px; 
      background-color: #3b82f6; color: white; font-size: 14px; font-weight: bold; 
      border: none; border-radius: 8px; margin: 5px; cursor: pointer; transition: 0.2s; 
    }
    .btn:active { background-color: #2563eb; transform: scale(0.95); }
    .btn-stop { background-color: #ef4444; width: 230px; height: 70px; font-size: 18px; margin-top: 15px; }
    .btn-stop:active { background-color: #dc2626; }
    .section { 
      margin: 15px auto; padding: 15px; max-width: 400px;
      border: 1px solid #333; border-radius: 10px; background-color: #2a2a2a; 
    }
    h3 { margin-top: 0; color: #38bdf8; font-size: 16px; }
  </style>
  <script>
    function cmd(action) {
      fetch('/' + action);
    }
    function updateSpeed(val) {
      document.getElementById('speedLabel').innerText = val;
      fetch('/speed?v=' + val);
    }
  </script>
</head>
<body>
  <h2>Motor Diagnostic Test</h2>
  <p>Test individual tires & movements at slow speeds</p>
  
  <div class="section">
    <label>Test Speed (PWM 50-255): <span id="speedLabel" style="color:#4ade80; font-weight:bold;">90</span></label><br><br>
    <input type="range" min="50" max="255" value="90" style="width: 80%" onchange="updateSpeed(this.value)">
  </div>

  <div class="section">
    <h3>All Wheels (Standard)</h3>
    <button class="btn" onmousedown="cmd('forward')" onmouseup="cmd('stop')" ontouchstart="cmd('forward')" ontouchend="cmd('stop')">FORWARD</button><br>
    <button class="btn" onmousedown="cmd('left')" onmouseup="cmd('stop')" ontouchstart="cmd('left')" ontouchend="cmd('stop')">LEFT</button>
    <button class="btn" onmousedown="cmd('right')" onmouseup="cmd('stop')" ontouchstart="cmd('right')" ontouchend="cmd('stop')">RIGHT</button><br>
    <button class="btn" onmousedown="cmd('reverse')" onmouseup="cmd('stop')" ontouchstart="cmd('reverse')" ontouchend="cmd('stop')">REVERSE</button>
  </div>
  
  <div class="section">
    <h3>Individual Sides (Diagnostic)</h3>
    <button class="btn" onmousedown="cmd('left_fwd')" onmouseup="cmd('stop')" ontouchstart="cmd('left_fwd')" ontouchend="cmd('stop')">Left FWD</button>
    <button class="btn" onmousedown="cmd('right_fwd')" onmouseup="cmd('stop')" ontouchstart="cmd('right_fwd')" ontouchend="cmd('stop')">Right FWD</button><br>
    <button class="btn" onmousedown="cmd('left_rev')" onmouseup="cmd('stop')" ontouchstart="cmd('left_rev')" ontouchend="cmd('stop')">Left REV</button>
    <button class="btn" onmousedown="cmd('right_rev')" onmouseup="cmd('stop')" ontouchstart="cmd('right_rev')" ontouchend="cmd('stop')">Right REV</button>
  </div>

  <button class="btn btn-stop" onclick="cmd('stop')">EMERGENCY STOP</button>

</body>
</html>
)rawliteral";

// ============================================================
//  MOTOR LOGIC
// ============================================================
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

// Full Movements
void forward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void reverse() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}
void left() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void right() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

// Individual Side Movements (For testing specific motors/tires)
void leftFwd() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}
void rightFwd() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}
void leftRev() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}
void rightRev() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // Initialize motor pins
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  // Initialize PWM for speed control
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA, 5000, 8);
  ledcAttach(ENB, 5000, 8);
  ledcWrite(ENA, speedPWM);
  ledcWrite(ENB, speedPWM);
#else
  ledcSetup(0, 5000, 8); 
  ledcAttachPin(ENA, 0);
  ledcSetup(1, 5000, 8); 
  ledcAttachPin(ENB, 1);
  ledcWrite(0, speedPWM);
  ledcWrite(1, speedPWM);
#endif
  stopMotors();

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500); 
    Serial.print(".");
  }
  Serial.println("\n✅ Connected!");
  Serial.print("Open this IP in your browser: http://");
  Serial.println(WiFi.localIP());

  // Web routes
  server.on("/", []() { 
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "text/html", html_page); 
  });
  
  server.on("/stop", []() { stopMotors(); server.send(200); });
  
  server.on("/forward", []() { forward(); server.send(200); });
  server.on("/reverse", []() { reverse(); server.send(200); });
  server.on("/left", []() { left(); server.send(200); });
  server.on("/right", []() { right(); server.send(200); });
  
  server.on("/left_fwd", []() { leftFwd(); server.send(200); });
  server.on("/right_fwd", []() { rightFwd(); server.send(200); });
  server.on("/left_rev", []() { leftRev(); server.send(200); });
  server.on("/right_rev", []() { rightRev(); server.send(200); });

  server.on("/speed", []() {
    if (server.hasArg("v")) {
      speedPWM = server.arg("v").toInt();
      if(speedPWM < 0) speedPWM = 0;
      if(speedPWM > 255) speedPWM = 255;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
      ledcWrite(ENA, speedPWM);
      ledcWrite(ENB, speedPWM);
#else
      ledcWrite(0, speedPWM);
      ledcWrite(1, speedPWM);
#endif
    }
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "text/plain", "OK");
  });

  server.begin();
}

void loop() {
  server.handleClient();
}
