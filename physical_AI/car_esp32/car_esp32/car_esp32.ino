#include <WiFi.h>
#include <WebServer.h>

// ============================================================
//  WIFI SETTINGS
// ============================================================
const char* WIFI_SSID = "Muhammad's iphone";
const char* WIFI_PASS = "jaffar12";

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
  </style>
  <script>
    function sendCommand(cmd) {
      fetch('/' + cmd);
    }
  </script>
</head>
<body>
  <h2>ESP32 Rover Control</h2>
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

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // Init Pins
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);

  // Enable motors constantly (Full speed)
  digitalWrite(ENA, HIGH);
  digitalWrite(ENB, HIGH);
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
  server.on("/", []() { server.send(200, "text/html", html_page); });
  
  server.on("/forward", []() { forwardMotors(); server.send(200, "text/plain", "OK"); });
  server.on("/reverse", []() { reverseMotors(); server.send(200, "text/plain", "OK"); });
  server.on("/left",    []() { turnLeftMotors(); server.send(200, "text/plain", "OK"); });
  server.on("/right",   []() { turnRightMotors(); server.send(200, "text/plain", "OK"); });
  server.on("/stop",    []() { stopMotors(); server.send(200, "text/plain", "OK"); });

  server.begin();
  Serial.println("Dashboard server started.");
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  server.handleClient();
}