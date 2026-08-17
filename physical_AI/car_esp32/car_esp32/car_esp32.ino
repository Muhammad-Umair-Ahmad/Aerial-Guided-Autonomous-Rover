#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>

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

// Sequence versioning
int sequenceVersion = 0;

// Preferences for persistent storage
Preferences preferences;

// ============================================================
//  CALIBRATION MODE — Turn Time Mapper (Phase 1)
//  Non-blocking state machine that tests left/right turns
//  at multiple PWM levels and logs exact timings to Serial.
// ============================================================
enum CalPhase {
  CAL_IDLE,
  CAL_STARTING,
  CAL_PRE_PAUSE,
  CAL_TURNING,
  CAL_POST_PAUSE,
  CAL_COMPLETE
};

CalPhase calPhase = CAL_IDLE;
int calStep = 0;
unsigned long calPhaseStart = 0;
int calSavedSpeed = 140;

// PWM levels to test (user-specified: 150, 160, 180, 200)
const int CAL_SPEEDS[] = {150, 160, 180, 200};
const int CAL_NUM_SPEEDS = 4;

// Starting test durations per speed (ms) — slower speeds get longer turns
const int CAL_DURATIONS[] = {1200, 1000, 800, 600};

// Total tests: each speed × 2 directions (left + right)
const int CAL_TOTAL_STEPS = CAL_NUM_SPEEDS * 2;

// Pause between tests (ms) — gives rover time to settle
const unsigned long CAL_PAUSE_MS = 2500;

// Results storage
struct CalResult {
  int pwm;
  unsigned long leftDuration;
  unsigned long rightDuration;
};
CalResult calResults[4];

// Single test tracking (for /calibrate/single endpoint)
bool singleTestActive = false;
unsigned long singleTestStart = 0;
int singleTestPWM = 0;
String singleTestDir = "";
unsigned long singleTestDuration = 0;

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

// ============================================================
//  SPEED HELPER — Centralised PWM setter
// ============================================================
void setMotorSpeed(int speed) {
  currentSpeed = speed;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(ENA, currentSpeed);
  ledcWrite(ENB, currentSpeed);
#else
  ledcWrite(0, currentSpeed);
  ledcWrite(1, currentSpeed);
#endif
}

// ============================================================
//  SEQUENCE MODE — Hardcoded Path Executor (Phase 2)
// ============================================================
struct Movement {
    int pwm;
    String direction;
    unsigned long duration;
};

const int MAX_MOVEMENTS = 50;
Movement movements[MAX_MOVEMENTS];
int movementCount = 0;

bool seqActive = false;
bool seqPaused = false;
int seqIndex = 0;
unsigned long seqStepStart = 0;
unsigned long seqRemainingMs = 0;

// ============================================================
//  PERSISTENT SEQUENCE STORAGE (NVS)
// ============================================================
void saveSequenceToNVS() {
    preferences.begin("seq", false); // read-write
    preferences.putInt("count", movementCount);
    preferences.putInt("version", sequenceVersion);
    for (int i = 0; i < movementCount && i < MAX_MOVEMENTS; i++) {
        String keyPwm = "p" + String(i);
        String keyDir = "d" + String(i);
        String keyDur = "t" + String(i);
        preferences.putInt(keyPwm.c_str(), movements[i].pwm);
        preferences.putString(keyDir.c_str(), movements[i].direction);
        preferences.putULong(keyDur.c_str(), movements[i].duration);
    }
    preferences.end();
    Serial.printf("[NVS] Saved %d movements (v%d)\n", movementCount, sequenceVersion);
}

void loadSequenceFromNVS() {
    preferences.begin("seq", true); // read-only
    int savedCount = preferences.getInt("count", 0);
    int savedVersion = preferences.getInt("version", 0);
    if (savedCount > 0 && savedCount <= MAX_MOVEMENTS) {
        movementCount = savedCount;
        sequenceVersion = savedVersion;
        for (int i = 0; i < movementCount; i++) {
            String keyPwm = "p" + String(i);
            String keyDir = "d" + String(i);
            String keyDur = "t" + String(i);
            movements[i].pwm = preferences.getInt(keyPwm.c_str(), 150);
            movements[i].direction = preferences.getString(keyDir.c_str(), "FORWARD");
            movements[i].duration = preferences.getULong(keyDur.c_str(), 1000);
        }
        Serial.printf("[NVS] Loaded %d movements (v%d)\n", movementCount, sequenceVersion);
    } else {
        Serial.println("[NVS] No saved sequence found.");
    }
    preferences.end();
}

// Simple checksum for verification
String computeChecksum() {
    unsigned long hash = 5381;
    for (int i = 0; i < movementCount; i++) {
        hash = hash * 33 + movements[i].pwm;
        for (unsigned int c = 0; c < movements[i].direction.length(); c++) {
            hash = hash * 33 + movements[i].direction[c];
        }
        hash = hash * 33 + movements[i].duration;
    }
    char buf[8];
    snprintf(buf, sizeof(buf), "%06lx", hash & 0xFFFFFF);
    return String(buf);
}

void startSequenceStep() {
    if (seqIndex >= movementCount) {
        seqActive = false;
        stopMotors();
        Serial.println("Sequence complete!");
        return;
    }
    
    Movement m = movements[seqIndex];
    setMotorSpeed(m.pwm);
    
    if (m.direction == "FORWARD") forwardMotors();
    else if (m.direction == "REVERSE") reverseMotors();
    else if (m.direction == "LEFT") turnLeftMotors();
    else if (m.direction == "RIGHT") turnRightMotors();
    
    seqStepStart = millis();
    if (!seqPaused) {
      seqRemainingMs = m.duration;
    }
    seqPaused = false;
    
    Serial.printf("SEQ STEP %d: %s at PWM %d for %lums\n", seqIndex, m.direction.c_str(), m.pwm, seqRemainingMs);
}

void runSequence() {
    if (!seqActive || seqPaused) return;

    unsigned long currentMillis = millis();
    unsigned long elapsed = currentMillis - seqStepStart;
    
    if (elapsed >= seqRemainingMs) {
        seqIndex++;
        seqPaused = false;
        startSequenceStep();
    }
}

// ============================================================
//  CALIBRATION STATE MACHINE
//  Runs non-blocking inside loop(). Each call checks the
//  current phase, advances when time thresholds are met.
// ============================================================
void runCalibration() {
  unsigned long now = millis();

  switch (calPhase) {

    // ---- Print header, reset step counter ----
    case CAL_STARTING: {
      Serial.println();
      Serial.println("========================================");
      Serial.println("  TURN CALIBRATION - TIME MAPPER");
      Serial.println("========================================");
      Serial.printf("  Speed levels : %d\n", CAL_NUM_SPEEDS);
      Serial.printf("  Total tests  : %d (left + right each)\n", CAL_TOTAL_STEPS);
      Serial.printf("  Pause between: %lums\n", CAL_PAUSE_MS);
      Serial.println("----------------------------------------");
      Serial.println("  PWM levels: 150, 160, 180, 200");
      Serial.println("  Durations : 1200, 1000, 800, 600 ms");
      Serial.println("========================================");
      Serial.println();
      calStep = 0;
      calPhase = CAL_PRE_PAUSE;
      calPhaseStart = now;
      break;
    }

    // ---- Wait before starting next turn ----
    case CAL_PRE_PAUSE: {
      if (now - calPhaseStart >= CAL_PAUSE_MS) {
        int speedIdx = calStep / 2;
        bool isLeft = (calStep % 2 == 0);
        int testSpeed = CAL_SPEEDS[speedIdx];
        unsigned long testDuration = CAL_DURATIONS[speedIdx];

        // Set PWM for this test
        setMotorSpeed(testSpeed);

        // Activate turn
        if (isLeft) {
          turnLeftMotors();
        } else {
          turnRightMotors();
        }

        Serial.printf(">> Test %d/%d | %-5s | PWM: %3d | Duration: %lums  ... ",
          calStep + 1, CAL_TOTAL_STEPS,
          isLeft ? "LEFT" : "RIGHT",
          testSpeed, testDuration);

        calPhaseStart = now;
        calPhase = CAL_TURNING;
      }
      break;
    }

    // ---- Motors are running — wait for duration to elapse ----
    case CAL_TURNING: {
      int speedIdx = calStep / 2;
      unsigned long testDuration = CAL_DURATIONS[speedIdx];

      if (now - calPhaseStart >= testDuration) {
        stopMotors();

        // Store result
        bool isLeft = (calStep % 2 == 0);
        calResults[speedIdx].pwm = CAL_SPEEDS[speedIdx];
        if (isLeft) {
          calResults[speedIdx].leftDuration = testDuration;
        } else {
          calResults[speedIdx].rightDuration = testDuration;
        }

        Serial.println("DONE");

        calStep++;
        if (calStep >= CAL_TOTAL_STEPS) {
          calPhase = CAL_COMPLETE;
        } else {
          calPhase = CAL_POST_PAUSE;
          calPhaseStart = now;
        }
      }
      break;
    }

    // ---- Pause after a turn before starting the next one ----
    case CAL_POST_PAUSE: {
      if (now - calPhaseStart >= CAL_PAUSE_MS) {
        calPhase = CAL_PRE_PAUSE;
        calPhaseStart = now;
      }
      break;
    }

    // ---- All tests done — print summary table ----
    case CAL_COMPLETE: {
      Serial.println();
      Serial.println("========================================");
      Serial.println("  CALIBRATION COMPLETE - RESULTS");
      Serial.println("========================================");
      Serial.println("  PWM  |  LEFT (ms)  |  RIGHT (ms)");
      Serial.println("  -----|-------------|-------------");
      for (int i = 0; i < CAL_NUM_SPEEDS; i++) {
        Serial.printf("  %3d  |  %7lu    |  %7lu\n",
          calResults[i].pwm,
          calResults[i].leftDuration,
          calResults[i].rightDuration);
      }
      Serial.println("========================================");
      Serial.println();
      Serial.println("  NEXT STEPS:");
      Serial.println("  1. Note which turns looked like ~90 deg");
      Serial.println("  2. Fine-tune with:");
      Serial.println("     /calibrate/single?dir=left&pwm=150&ms=XXX");
      Serial.println("  3. Repeat until you nail exact 90 deg times");
      Serial.println("  4. Record final mapping for lawnmower sequence");
      Serial.println("========================================");
      Serial.println();

      // Restore original speed
      setMotorSpeed(calSavedSpeed);
      calPhase = CAL_IDLE;
      break;
    }

    default:
      break;
  }
}

// ============================================================
//  CORS & COMMAND HELPERS
// ============================================================
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

  // Load saved sequence from persistent storage
  loadSequenceFromNVS();

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

  // ============================================================
  //  CALIBRATION ENDPOINTS
  // ============================================================

  // Start full calibration sweep (4 speeds × 2 directions = 8 tests)
  server.on("/calibrate", []() {
    sendCORSHeader();
    if (calPhase != CAL_IDLE) {
      server.send(409, "text/plain", "Calibration already running. Hit /calibrate/stop first.");
      return;
    }
    calSavedSpeed = currentSpeed;
    calPhase = CAL_STARTING;
    calPhaseStart = millis();
    lastCommandTime = millis(); // prevent watchdog during calibration
    server.send(200, "text/plain", "Calibration started. Watch Serial Monitor for results.");
  });

  // Emergency abort calibration
  server.on("/calibrate/stop", []() {
    sendCORSHeader();
    stopMotors();
    calPhase = CAL_IDLE;
    setMotorSpeed(calSavedSpeed);
    singleTestActive = false;
    isTimedMove = false;
    lastCommand = "stop";
    Serial.println("\n!! CALIBRATION ABORTED BY USER !!\n");
    server.send(200, "text/plain", "Calibration aborted. Motors stopped.");
  });

  // Check calibration progress (JSON)
  server.on("/calibrate/status", []() {
    sendCORSHeader();
    String phase;
    if (calPhase == CAL_IDLE) phase = "idle";
    else if (calPhase == CAL_TURNING) phase = "turning";
    else if (calPhase == CAL_COMPLETE) phase = "complete";
    else phase = "paused";

    String json = "{";
    json += "\"running\":" + String(calPhase != CAL_IDLE ? "true" : "false") + ",";
    json += "\"step\":" + String(calStep) + ",";
    json += "\"totalSteps\":" + String(CAL_TOTAL_STEPS) + ",";
    json += "\"phase\":\"" + phase + "\"";
    json += "}";
    server.send(200, "application/json", json);
  });

  // Single test: /calibrate/single?dir=left&pwm=150&ms=1000
  // Use this to fine-tune a specific speed+direction+duration combo
  server.on("/calibrate/single", []() {
    sendCORSHeader();
    if (calPhase != CAL_IDLE) {
      server.send(409, "text/plain", "Full calibration running. Hit /calibrate/stop first.");
      return;
    }

    String dir = server.hasArg("dir") ? server.arg("dir") : "left";
    int pwm = server.hasArg("pwm") ? server.arg("pwm").toInt() : currentSpeed;
    int ms  = server.hasArg("ms")  ? server.arg("ms").toInt()  : 1000;

    // Safety clamps
    if (pwm < 0) pwm = 0;
    if (pwm > 255) pwm = 255;
    if (ms < 100) ms = 100;
    if (ms > 5000) ms = 5000;

    // Save current speed, apply test speed
    calSavedSpeed = currentSpeed;
    setMotorSpeed(pwm);

    // Start turn
    if (dir == "right") {
      turnRightMotors();
    } else {
      turnLeftMotors();
      dir = "left"; // normalise
    }

    // Use existing timed-move mechanism
    stopTime = millis() + ms;
    isTimedMove = true;
    lastCommandTime = millis(); // prevent watchdog
    lastCommand = dir;

    // Track for detailed logging on completion
    singleTestActive = true;
    singleTestStart = millis();
    singleTestPWM = pwm;
    singleTestDir = dir;
    singleTestDuration = ms;

    Serial.println();
    Serial.println("----------------------------------------");
    Serial.printf(">> SINGLE TEST | %-5s | PWM: %3d | Duration: %dms\n",
      dir.c_str(), pwm, ms);

    server.send(200, "text/plain",
      "Single test: " + dir + " @ PWM " + String(pwm) + " for " + String(ms) + "ms");
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

  // Sequence Mode Endpoints
  server.on("/sequence/start", []() {
    sendCORSHeader();
    if (seqActive && !seqPaused) {
       server.send(409, "text/plain", "Sequence already running.");
       return;
    }
    if (!seqPaused || seqIndex >= movementCount) {
        seqIndex = 0;
        seqPaused = false;
    }
    seqActive = true;
    startSequenceStep();
    server.send(200, "text/plain", "Sequence started.");
  });

  server.on("/sequence/pause", []() {
    sendCORSHeader();
    if (seqActive && !seqPaused) {
        seqPaused = true;
        stopMotors();
        unsigned long elapsed = millis() - seqStepStart;
        if (elapsed < seqRemainingMs) {
            seqRemainingMs -= elapsed;
        } else {
            seqRemainingMs = 0;
        }
        Serial.println("Sequence paused.");
    }
    server.send(200, "text/plain", "Sequence paused.");
  });

  server.on("/sequence/resume", []() {
    sendCORSHeader();
    if (seqActive && seqPaused) {
        seqActive = true;
        startSequenceStep();
        server.send(200, "text/plain", "Sequence resumed.");
    } else {
        server.send(400, "text/plain", "Sequence not paused.");
    }
  });

  server.on("/sequence/stop", []() {
    sendCORSHeader();
    seqActive = false;
    seqPaused = false;
    stopMotors();
    Serial.println("Sequence stopped.");
    server.send(200, "text/plain", "Sequence stopped.");
  });

  server.on("/sequence/upload", HTTP_POST, []() {
    sendCORSHeader();
    if (!server.hasArg("plain")) {
      server.send(400, "application/json", "{\"ok\":false,\"error\":\"No sequence data provided\"}");
      return;
    }
    String body = server.arg("plain");
    
    // Parse into temporary buffer for validation
    Movement tempMovements[MAX_MOVEMENTS];
    int tempCount = 0;
    int startIndex = 0;
    bool parseError = false;
    
    while (startIndex < body.length() && tempCount < MAX_MOVEMENTS) {
      int endIndex = body.indexOf(';', startIndex);
      if (endIndex == -1) endIndex = body.length();
      
      String row = body.substring(startIndex, endIndex);
      row.trim();
      if (row.length() > 0) {
        int comma1 = row.indexOf(',');
        int comma2 = row.lastIndexOf(',');
        if (comma1 != -1 && comma2 != -1 && comma1 != comma2) {
           int pwm = row.substring(0, comma1).toInt();
           String dir = row.substring(comma1 + 1, comma2);
           dir.trim();
           dir.toUpperCase();
           unsigned long dur = row.substring(comma2 + 1).toInt();
           
           // Validate
           if (pwm < 0 || pwm > 255) { parseError = true; break; }
           if (dir != "FORWARD" && dir != "REVERSE" && dir != "LEFT" && dir != "RIGHT") { parseError = true; break; }
           if (dur == 0 || dur > 30000) { parseError = true; break; }
           
           tempMovements[tempCount].pwm = pwm;
           tempMovements[tempCount].direction = dir;
           tempMovements[tempCount].duration = dur;
           tempCount++;
        }
      }
      startIndex = endIndex + 1;
    }
    
    if (parseError || tempCount == 0) {
      server.send(400, "application/json", "{\"ok\":false,\"error\":\"Invalid sequence data\"}");
      return;
    }
    
    // Stop any running sequence before replacing
    seqActive = false;
    seqPaused = false;
    seqIndex = 0;
    stopMotors();
    
    // Copy validated data to active buffer
    movementCount = tempCount;
    for (int i = 0; i < tempCount; i++) {
      movements[i] = tempMovements[i];
    }
    
    // Increment version
    sequenceVersion++;
    
    // Save to persistent storage
    saveSequenceToNVS();
    
    // Compute checksum
    String checksum = computeChecksum();
    
    Serial.printf("Uploaded %d movements (v%d, checksum=%s)\n", movementCount, sequenceVersion, checksum.c_str());
    
    // Return JSON with full details
    String json = "{";
    json += "\"ok\":true,";
    json += "\"steps\":" + String(movementCount) + ",";
    json += "\"version\":" + String(sequenceVersion) + ",";
    json += "\"checksum\":\"" + checksum + "\"";
    json += "}";
    server.send(200, "application/json", json);
  });

  // Sequence status — returns currently stored sequence details
  server.on("/sequence/status", []() {
    sendCORSHeader();
    String checksum = computeChecksum();
    String json = "{";
    json += "\"step_count\":" + String(movementCount) + ",";
    json += "\"version\":" + String(sequenceVersion) + ",";
    json += "\"checksum\":\"" + checksum + "\",";
    json += "\"steps\":[";
    for (int i = 0; i < movementCount; i++) {
      if (i > 0) json += ",";
      json += "{\"pwm\":" + String(movements[i].pwm);
      json += ",\"dir\":\"" + movements[i].direction + "\"";
      json += ",\"ms\":" + String(movements[i].duration) + "}";
    }
    json += "]}";
    server.send(200, "application/json", json);
  });

  server.begin();
  Serial.println("Dashboard server started.");
  Serial.println("Calibration endpoints ready:");
  Serial.println("  /calibrate        — Run full sweep");
  Serial.println("  /calibrate/stop   — Abort");
  Serial.println("  /calibrate/status — Check progress");
  Serial.println("  /calibrate/single — Fine-tune one test");
  Serial.println();
  lastCommandTime = millis();
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
  server.handleClient();

  unsigned long currentMillis = millis();

  // ---- Sequence state machine ----
  if (seqActive && !seqPaused) {
    runSequence();
    lastCommandTime = currentMillis;
    return;
  }

  // ---- Calibration state machine takes priority ----
  if (calPhase != CAL_IDLE) {
    runCalibration();
    // Suppress watchdog during calibration to avoid auto-stopping test turns
    lastCommandTime = currentMillis;
    return;
  }

  // Watchdog Auto-Stop: if no command received for 5000ms
  if (currentMillis - lastCommandTime > 5000 && lastCommand != "stop") {
    stopMotors();
    lastCommand = "stop";
    isTimedMove = false;
    singleTestActive = false;
    Serial.println("Watchdog: Auto-stopped motors.");
  }

  // Timed Movement: if moving for a set duration and time is up
  if (isTimedMove && currentMillis >= stopTime) {
    stopMotors();
    lastCommand = "stop";
    isTimedMove = false;

    // If this was a single calibration test, log the detailed result
    if (singleTestActive) {
      unsigned long elapsed = currentMillis - singleTestStart;
      Serial.printf("   DONE | %-5s | PWM: %3d | Ran for: %lums\n",
        singleTestDir.c_str(), singleTestPWM, elapsed);
      Serial.println("   Was that ~90 deg? If not, adjust ms and try again.");
      Serial.println("----------------------------------------");
      Serial.println();
      singleTestActive = false;
      // Restore original speed
      setMotorSpeed(calSavedSpeed);
    } else {
      Serial.println("Timed move: Auto-stopped motors.");
    }
  }
}