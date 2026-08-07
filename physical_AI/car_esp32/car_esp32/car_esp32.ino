#include <WiFi.h>
#include <WebServer.h>

// ============================================================
//  WIFI SETTINGS
// ============================================================
const char* WIFI_SSID = "laptop";
const char* WIFI_PASS = "12345678";

WebServer server(80);

// ============================================================
//  HARDWARE PINS (Safe ESP32 Pins)
// ============================================================
#define IN1 26
#define IN2 27
#define IN3 14
#define IN4 13
#define ENA 32
#define ENB 33

// ============================================================
//  STATE VARIABLES
// ============================================================
int currentSpeed = 140;
String lastCommand = "stop";
unsigned long lastCommandTime = 0;
unsigned long stopTime = 0;
bool isTimedMove = false;

// ============================================================
//  HTML DASHBOARD (Mobile Friendly UI)
// ============================================================
const char* html_page = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <title>ESP32 Car Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; text-align: center; background-color: #222; color: white; margin-top: 50px; }
    .btn { display: inline-block; width: 80px; height: 80px; background-color: #4CAF50; color: white; font-size: 24px; font-weight: bold; border: none; border-radius: 10px; margin: 10px; cursor: pointer; }
    .btn:active { background-color: #3e8e41; }
    .btn-stop { background-color: #f44336; }
    .btn-stop:active { background-color: #da190b; }
    .grid { display: grid; grid-template-columns: repeat(3, 100px); justify-content: center; gap: 10px; margin-top: 30px; }
    .empty { width: 80px; height: 80px; }
    .slider-container { margin-top: 20px; }
    input[type=range] { width: 80%; max-width: 300px; }
  </style>
  <script>
    function sendCommand(cmd) {
      fetch('/' + cmd);
    }
    function updateSpeed(val) {
      document.getElementById('speedVal').innerText = val;
    }
    function setSpeed(val) {
      fetch('/speed?v=' + val);
    }
  </script>
</head>
<body>
  <h2>ESP32 Rover Control</h2>
  <div class="slider-container">
    <label for="speedSlider">Speed: <span id="speedVal">140</span></label><br>
    <input type="range" id="speedSlider" min="0" max="255" value="140" oninput="updateSpeed(this.value)" onchange="setSpeed(this.value)">
  </div>
  <div class="grid">
    <div class="empty"></div>
    <button class="btn" onmousedown="sendCommand('forward')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('forward')" ontouchend="sendCommand('stop')">W</button>
    <div class="empty"></div>
    
    <button class="btn" onmousedown="sendCommand('left')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('left')" ontouchend="sendCommand('stop')">A</button>
    <button class="btn btn-stop" onclick="sendCommand('stop')">STOP</button>
    <button class="btn" onmousedown="sendCommand('right')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('right')" ontouchend="sendCommand('stop')">D</button>
    
    <div class="empty"></div>
    <button class="btn" onmousedown="sendCommand('reverse')" onmouseup="sendCommand('stop')" ontouchstart="sendCommand('reverse')" ontouchend="sendCommand('stop')">S</button>
    <div class="empty"></div>
  </div>
</body>
</html>
)rawliteral";

// ============================================================
//  MOTOR HELPERS
// ============================================================
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void forwardMotors() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void reverseMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void turnLeftMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void turnRightMotors() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void sendCORSHeader() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
}

void handleCommand(String cmd) {
  sendCORSHeader();
  lastCommand = cmd;
  lastCommandTime = millis();
  
  if (server.hasArg("ms")) {
    stopTime = millis() + server.arg("ms").toInt();
    isTimedMove = true;
  } else {
    isTimedMove = false;
  }

  if (cmd == "forward") forwardMotors();
  else if (cmd == "reverse") reverseMotors();
  else if (cmd == "left") turnLeftMotors();
  else if (cmd == "right") turnRightMotors();
  else if (cmd == "stop") {
    stopMotors();
    isTimedMove = false;
  }
  
  server.send(200, "text/plain", "OK");
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // Init Pins
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  // Configure PWM for speed control
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(ENA, 5000, 8);
  ledcAttach(ENB, 5000, 8);
  ledcWrite(ENA, currentSpeed);
  ledcWrite(ENB, currentSpeed);
#else
  ledcSetup(0, 5000, 8); // Channel 0, 5kHz, 8-bit resolution
  ledcAttachPin(ENA, 0);
  ledcSetup(1, 5000, 8); // Channel 1, 5kHz, 8-bit resolution
  ledcAttachPin(ENB, 1);
  
  // Set default speed
  ledcWrite(0, currentSpeed);
  ledcWrite(1, currentSpeed);
#endif
  stopMotors();

  // Connect to WiFi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("\nConnecting to Hotspot");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n✅ Connected!");
  Serial.print("Open this IP in your browser: http://");
  Serial.println(WiFi.localIP());

  // Define Web Routes
  server.on("/", []() { 
    sendCORSHeader(); 
    server.send(200, "text/html", html_page); 
  });
  
  server.on("/forward", []() { handleCommand("forward"); });
  server.on("/reverse", []() { handleCommand("reverse"); });
  server.on("/left",    []() { handleCommand("left"); });
  server.on("/right",   []() { handleCommand("right"); });
  server.on("/stop",    []() { handleCommand("stop"); });

  server.on("/speed", []() {
    sendCORSHeader();
    if (server.hasArg("v")) {
      currentSpeed = server.arg("v").toInt();
      // Constrain speed to 0-255
      if (currentSpeed < 0) currentSpeed = 0;
      if (currentSpeed > 255) currentSpeed = 255;
      
#if ESP_ARDUINO_VERSION_MAJOR >= 3
      ledcWrite(ENA, currentSpeed);
      ledcWrite(ENB, currentSpeed);
#else
      ledcWrite(0, currentSpeed);
      ledcWrite(1, currentSpeed);
#endif
    }
    server.send(200, "text/plain", String(currentSpeed));
  });

  server.on("/status", []() {
    sendCORSHeader();
    String json = "{";
    json += "\"alive\":true,";
    json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
    json += "\"speed\":" + String(currentSpeed) + ",";
    json += "\"uptime\":" + String(millis()) + ",";
    json += "\"last_command\":\"" + lastCommand + "\"";
    json += "}";
    server.send(200, "application/json", json);
  });

  // Handle OPTIONS for CORS preflight
  server.onNotFound([]() {
    if (server.method() == HTTP_OPTIONS) {
      server.sendHeader("Access-Control-Allow-Origin", "*");
      server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
      server.sendHeader("Access-Control-Allow-Headers", "*");
      server.send(204);
    } else {
      sendCORSHeader();
      server.send(404, "text/plain", "Not Found");
    }
  });

  server.begin();
  Serial.println("Dashboard server started.");
  lastCommandTime = millis();
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  server.handleClient();

  unsigned long currentMillis = millis();

  // Watchdog Auto-Stop: if no command received for 5000ms
  if (currentMillis - lastCommandTime > 5000 && lastCommand != "stop") {
    stopMotors();
    lastCommand = "stop";
    isTimedMove = false;
    Serial.println("Watchdog: Auto-stopped motors.");
  }

  // Timed Movement: if moving for a set duration and time is up
  if (isTimedMove && currentMillis >= stopTime) {
    stopMotors();
    lastCommand = "stop";
    isTimedMove = false;
    Serial.println("Timed move: Auto-stopped motors.");
  }
}