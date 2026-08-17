"""
Phase 1: Turn Time Mapper — Calibration Dashboard
===================================================
Run this script to launch the calibration dashboard in your browser.

Usage:
    python phase1_calibration.py

Workflow:
    1. Select a PWM speed and direction on the dashboard
    2. Press START (or SPACEBAR) — the rover starts turning, timer starts
    3. Watch the rover physically
    4. Press STOP (or SPACEBAR) when it reaches ~90 degrees
    5. The exact time is recorded automatically
    6. Repeat for all speed/direction combos
    7. Copy the final mapping for Phase 2

No camera needed. No Serial Monitor needed. Everything is on the dashboard.
"""

import http.server
import webbrowser
import threading
import sys
import os

# ============================================================
#  CONFIGURATION
# ============================================================
DASHBOARD_PORT = 5500

# ============================================================
#  HTML DASHBOARD (embedded — zero dependencies)
# ============================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phase 1 — Turn Time Mapper</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <style>
    /* ====== RESET & BASE ====== */
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: #080b16;
      color: #e2e8f0;
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* Subtle grid background */
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(34, 211, 238, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34, 211, 238, 0.03) 1px, transparent 1px);
      background-size: 40px 40px;
      pointer-events: none;
      z-index: 0;
    }

    .container {
      max-width: 880px;
      margin: 0 auto;
      padding: 24px 16px 120px;
      position: relative;
      z-index: 1;
    }

    /* ====== HEADER ====== */
    header {
      text-align: center;
      margin-bottom: 28px;
    }

    header h1 {
      font-size: 28px;
      font-weight: 800;
      background: linear-gradient(135deg, #22d3ee, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: -0.5px;
    }

    header .subtitle {
      font-size: 14px;
      color: #64748b;
      margin-top: 4px;
      font-weight: 500;
    }

    .status-bar {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-top: 14px;
      font-size: 13px;
      color: #94a3b8;
      font-weight: 500;
    }

    .ip-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    .ip-input {
      padding: 10px 16px;
      border: 1px solid #334155;
      border-radius: 10px;
      background: #1e293b;
      color: #e2e8f0;
      font-family: 'JetBrains Mono', monospace;
      font-size: 15px;
      width: 210px;
      text-align: center;
      outline: none;
      transition: border-color 0.2s;
    }

    .ip-input:focus {
      border-color: #22d3ee;
      box-shadow: 0 0 12px rgba(34, 211, 238, 0.15);
    }

    .ip-input::placeholder {
      color: #475569;
      font-size: 13px;
    }

    .connect-btn {
      padding: 10px 22px;
      border: 1px solid #22d3ee;
      border-radius: 10px;
      background: rgba(34, 211, 238, 0.1);
      color: #22d3ee;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      letter-spacing: 0.3px;
    }

    .connect-btn:hover {
      background: rgba(34, 211, 238, 0.2);
      box-shadow: 0 0 15px rgba(34, 211, 238, 0.15);
    }

    .connect-btn:active {
      transform: scale(0.96);
    }

    .connect-btn.connected {
      border-color: #10b981;
      background: rgba(16, 185, 129, 0.1);
      color: #10b981;
    }

    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #475569;
      transition: background 0.3s;
    }

    .status-dot.connected {
      background: #10b981;
      box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
      animation: dotPulse 2s ease-in-out infinite;
    }

    .status-dot.disconnected {
      background: #ef4444;
      box-shadow: 0 0 8px rgba(239, 68, 68, 0.4);
    }

    @keyframes dotPulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    /* ====== CARDS ====== */
    .card {
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid rgba(51, 65, 85, 0.5);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 20px;
      backdrop-filter: blur(8px);
      transition: border-color 0.3s;
    }

    .card h3 {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: #94a3b8;
      margin-bottom: 18px;
    }

    .card h3 small {
      font-weight: 500;
      text-transform: none;
      letter-spacing: 0;
      color: #64748b;
    }

    /* Active card border glow during turn */
    .card.turning {
      border-color: rgba(34, 211, 238, 0.4);
      box-shadow: 0 0 30px rgba(34, 211, 238, 0.08);
    }

    /* ====== SELECTOR BUTTONS ====== */
    .selector-row {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }

    .selector-label {
      font-size: 13px;
      font-weight: 600;
      color: #94a3b8;
      min-width: 65px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .btn-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    .sel-btn {
      padding: 10px 20px;
      border: 1px solid #334155;
      border-radius: 10px;
      background: #1e293b;
      color: #94a3b8;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      min-width: 60px;
      text-align: center;
    }

    .sel-btn:hover {
      border-color: #475569;
      background: #263044;
      color: #e2e8f0;
    }

    .sel-btn.active {
      border-color: #22d3ee;
      background: rgba(34, 211, 238, 0.1);
      color: #22d3ee;
      box-shadow: 0 0 12px rgba(34, 211, 238, 0.15);
    }

    .sel-btn:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .custom-slider {
      -webkit-appearance: none;
      width: 150px;
      height: 6px;
      background: #1e293b;
      border-radius: 4px;
      outline: none;
    }
    .custom-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: #22d3ee;
      cursor: pointer;
      box-shadow: 0 0 8px rgba(34, 211, 238, 0.4);
    }

    /* ====== TIMER DISPLAY ====== */
    .timer-area {
      text-align: center;
      padding: 28px 0 20px;
    }

    .timer-state {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 2px;
      margin-bottom: 8px;
      color: #475569;
      transition: color 0.3s;
    }

    .timer-state.running {
      color: #22d3ee;
      animation: stateBlink 1s ease-in-out infinite;
    }

    @keyframes stateBlink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .timer-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 80px;
      font-weight: 700;
      color: #22d3ee;
      letter-spacing: 2px;
      line-height: 1;
      transition: text-shadow 0.3s;
    }

    .timer-value.running {
      animation: timerGlow 1.5s ease-in-out infinite;
    }

    @keyframes timerGlow {
      0%, 100% { text-shadow: 0 0 20px rgba(34, 211, 238, 0.2); }
      50% { text-shadow: 0 0 40px rgba(34, 211, 238, 0.5); }
    }

    .timer-unit {
      font-family: 'JetBrains Mono', monospace;
      font-size: 28px;
      color: #475569;
      margin-left: 4px;
      font-weight: 500;
    }

    .last-recorded {
      font-size: 14px;
      color: #10b981;
      font-weight: 600;
      margin-top: 10px;
      min-height: 22px;
      transition: opacity 0.3s;
    }

    /* ====== ACTION BUTTONS ====== */
    .action-row {
      display: flex;
      gap: 12px;
      justify-content: center;
      margin-top: 8px;
    }

    .action-btn {
      flex: 1;
      max-width: 220px;
      padding: 16px 24px;
      border: none;
      border-radius: 12px;
      font-family: 'Inter', sans-serif;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.2s;
      letter-spacing: 0.5px;
    }

    .action-btn:active {
      transform: scale(0.97);
    }

    .start-btn {
      background: linear-gradient(135deg, #10b981, #059669);
      color: #ecfdf5;
      box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }

    .start-btn:hover:not(:disabled) {
      box-shadow: 0 6px 25px rgba(16, 185, 129, 0.4);
      transform: translateY(-1px);
    }

    .stop-btn {
      background: linear-gradient(135deg, #ef4444, #dc2626);
      color: #fef2f2;
      box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }

    .stop-btn:hover:not(:disabled) {
      box-shadow: 0 6px 25px rgba(239, 68, 68, 0.4);
      transform: translateY(-1px);
    }

    .action-btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
      transform: none !important;
      box-shadow: none !important;
    }

    .hint {
      text-align: center;
      font-size: 12px;
      color: #475569;
      margin-top: 14px;
    }

    .hint kbd {
      padding: 2px 7px;
      border: 1px solid #334155;
      border-radius: 4px;
      background: #1e293b;
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      color: #94a3b8;
    }

    /* ====== QUICK TESTS ====== */
    .quick-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }

    .quick-btn {
      padding: 12px 16px;
      border: 1px solid #1e293b;
      border-radius: 10px;
      background: #111827;
      color: #94a3b8;
      font-family: 'Inter', sans-serif;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }

    .quick-btn:hover {
      border-color: #334155;
      background: #1e293b;
      color: #e2e8f0;
    }

    .quick-btn:active {
      transform: scale(0.97);
    }

    .quick-btn:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    .quick-btn .q-pwm {
      font-family: 'JetBrains Mono', monospace;
      color: #22d3ee;
      font-weight: 700;
    }

    .quick-btn .q-dir {
      font-size: 12px;
    }

    .quick-btn .q-check {
      color: #10b981;
      font-size: 14px;
    }

    /* ====== TABLES ====== */
    table {
      width: 100%;
      border-collapse: collapse;
    }

    thead th {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #64748b;
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #1e293b;
    }

    tbody td {
      padding: 10px 12px;
      font-size: 14px;
      border-bottom: 1px solid rgba(30, 41, 59, 0.5);
      font-weight: 500;
    }

    tbody tr:hover {
      background: rgba(30, 41, 59, 0.3);
    }

    tbody tr.just-added {
      animation: rowFlash 0.6s ease;
    }

    @keyframes rowFlash {
      0% { background: rgba(16, 185, 129, 0.2); }
      100% { background: transparent; }
    }

    .dir-forward { color: #10b981; }
    .dir-reverse { color: #ef4444; }
    .dir-left { color: #f59e0b; }
    .dir-right { color: #818cf8; }

    .time-cell {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: #22d3ee;
    }

    .has-value {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: #10b981;
    }

    .no-value {
      color: #334155;
    }

    .test-count {
      font-size: 11px;
      color: #64748b;
      font-weight: 400;
      font-family: 'Inter', sans-serif;
      margin-left: 4px;
    }

    .delete-btn {
      padding: 4px 10px;
      border: 1px solid #334155;
      border-radius: 6px;
      background: transparent;
      color: #64748b;
      cursor: pointer;
      font-size: 14px;
      transition: all 0.2s;
    }

    .delete-btn:hover {
      border-color: #ef4444;
      color: #ef4444;
      background: rgba(239, 68, 68, 0.1);
    }

    .empty-state {
      text-align: center;
      color: #334155;
      font-style: italic;
      padding: 24px !important;
    }

    /* ====== BOTTOM ACTIONS ====== */
    .card-actions {
      display: flex;
      gap: 10px;
      margin-top: 16px;
      flex-wrap: wrap;
    }

    .small-btn {
      padding: 8px 16px;
      border: 1px solid #334155;
      border-radius: 8px;
      background: #1e293b;
      color: #94a3b8;
      font-family: 'Inter', sans-serif;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }

    .small-btn:hover {
      border-color: #475569;
      color: #e2e8f0;
    }

    .small-btn.primary {
      border-color: #22d3ee;
      color: #22d3ee;
      background: rgba(34, 211, 238, 0.08);
    }

    .small-btn.primary:hover {
      background: rgba(34, 211, 238, 0.15);
    }

    .small-btn.danger:hover {
      border-color: #ef4444;
      color: #ef4444;
    }

    /* ====== EMERGENCY STOP ====== */
    .emergency-btn {
      position: fixed;
      bottom: 24px;
      right: 24px;
      padding: 16px 28px;
      border: 2px solid #dc2626;
      border-radius: 14px;
      background: rgba(153, 27, 27, 0.9);
      color: #fef2f2;
      font-family: 'Inter', sans-serif;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      z-index: 100;
      backdrop-filter: blur(8px);
      box-shadow: 0 4px 20px rgba(220, 38, 38, 0.3);
      transition: all 0.2s;
      letter-spacing: 0.3px;
    }

    .emergency-btn:hover {
      background: rgba(185, 28, 28, 0.95);
      box-shadow: 0 6px 30px rgba(220, 38, 38, 0.5);
      transform: translateY(-2px);
    }

    .emergency-btn:active {
      transform: scale(0.95);
    }

    /* ====== TOAST NOTIFICATIONS ====== */
    .toast {
      position: fixed;
      top: 20px;
      right: 20px;
      padding: 14px 24px;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      z-index: 200;
      transform: translateX(calc(100% + 30px));
      transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      max-width: 400px;
      backdrop-filter: blur(8px);
    }

    .toast.show { transform: translateX(0); }

    .toast-info {
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid #334155;
      color: #e2e8f0;
    }

    .toast-success {
      background: rgba(6, 78, 59, 0.9);
      border: 1px solid #10b981;
      color: #ecfdf5;
    }

    .toast-warning {
      background: rgba(120, 53, 15, 0.9);
      border: 1px solid #f59e0b;
      color: #fefce8;
    }

    /* ====== DIVIDER ====== */
    .divider {
      border: none;
      border-top: 1px solid #1e293b;
      margin: 20px 0;
    }

    /* ====== RESPONSIVE ====== */
    @media (max-width: 600px) {
      .timer-value { font-size: 56px; }
      .action-row { flex-direction: column; }
      .action-btn { max-width: 100%; }
      .quick-grid { grid-template-columns: repeat(2, 1fr); }
      .emergency-btn { bottom: 16px; right: 16px; left: 16px; text-align: center; }
    }
  </style>
</head>
<body>

  <div class="container">

    <!-- ====== HEADER ====== -->
    <header>
      <h1>Turn Time Mapper</h1>
      <p class="subtitle">Phase 1 Calibration Dashboard</p>
      <div class="ip-row">
        <input type="text" class="ip-input" id="ipInput" placeholder="ESP32 IP (e.g. 192.168.137.10)">
        <button class="connect-btn" id="connectBtn" onclick="connectToESP32()">Connect</button>
      </div>
      <div class="status-bar">
        <span class="status-dot" id="statusDot"></span>
        <span id="statusText">Enter the ESP32 IP address from Serial Monitor and press Connect</span>
      </div>
    </header>

    <!-- ====== MAIN CONTROLS ====== -->
    <div class="card" id="controlsCard">
      <div class="selector-row">
        <span class="selector-label">Speed</span>
        <div class="btn-group" id="pwmGroup">
          <!-- Dynamically populated -->
        </div>
      </div>

      <div class="selector-row">
        <span class="selector-label">Custom</span>
        <input type="range" id="pwmSlider" min="80" max="255" value="150" class="custom-slider" oninput="updateSliderVal()">
        <span id="sliderVal" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; width: 35px; text-align: right; color: #22d3ee;">150</span>
        <button class="small-btn primary" style="margin-left: 10px; padding: 6px 12px; font-size: 12px;" onclick="addCustomSpeed()">Save</button>
      </div>

      <div class="selector-row">
        <span class="selector-label">Turn</span>
        <div class="btn-group" id="dirGroup">
          <button class="sel-btn" data-dir="forward" onclick="selectDir('forward')">&#8593; FWD</button>
          <button class="sel-btn" data-dir="reverse" onclick="selectDir('reverse')">&#8595; REV</button>
          <button class="sel-btn active" data-dir="left" onclick="selectDir('left')">&#8592; LEFT</button>
          <button class="sel-btn" data-dir="right" onclick="selectDir('right')">RIGHT &#8594;</button>
        </div>
      </div>

      <div class="timer-area">
        <div class="timer-state" id="timerState">READY</div>
        <span class="timer-value" id="timerValue">0.000</span><span class="timer-unit">s</span>
        <div class="last-recorded" id="lastRecorded"></div>
      </div>

      <div class="action-row">
        <button class="action-btn start-btn" id="startBtn" onclick="startTurn()">&#9654; START TURN</button>
        <button class="action-btn stop-btn" id="stopBtn" onclick="stopAndRecord()" disabled>&#9632; STOP &amp; RECORD</button>
      </div>

      <p class="hint">Press <kbd>SPACE</kbd> to start / stop</p>
    </div>

    <!-- ====== QUICK TESTS ====== -->
    <div class="card">
      <h3>Quick Tests <small>&#8212; one click to start</small></h3>
      <div class="quick-grid" id="quickGrid"></div>
    </div>

    <!-- ====== MEASUREMENTS LOG ====== -->
    <div class="card">
      <h3>Measurements Log</h3>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>PWM</th>
            <th>Direction</th>
            <th>Time</th>
            <th style="width:50px"></th>
          </tr>
        </thead>
        <tbody id="logBody"></tbody>
      </table>
      <div class="card-actions">
        <button class="small-btn danger" onclick="clearAll()">Clear All</button>
      </div>
    </div>

    <!-- ====== FINAL MAPPING ====== -->
    <div class="card">
      <h3>Final Mapping <small>&#8212; latest measurement per combo (for Phase 2)</small></h3>
      <table>
        <thead>
          <tr>
            <th>PWM</th>
            <th>FWD (ms)</th>
            <th>REV (ms)</th>
            <th>LEFT 90&deg; (ms)</th>
            <th>RIGHT 90&deg; (ms)</th>
          </tr>
        </thead>
        <tbody id="mappingBody"></tbody>
      </table>
      <div class="card-actions">
        <button class="small-btn primary" onclick="copyMapping()">&#128203; Copy as C++ Array</button>
        <button class="small-btn" onclick="copyMappingText()">&#128203; Copy as Text</button>
      </div>
    </div>

  </div>

  <!-- ====== EMERGENCY STOP ====== -->
  <button class="emergency-btn" onclick="emergencyStop()">&#128721; EMERGENCY STOP</button>

<script>
// ============================================================
//  CONFIG
// ============================================================
let ESP32 = "";
let espIP = "";
let SPEEDS = [150, 160, 180, 200];

// ============================================================
//  STATE
// ============================================================
let selectedPWM = 150;
let selectedDir = "left";
let isRunning = false;
let timerStartTime = 0;
let animFrame = null;
let measurements = [];
let connected = false;

// ============================================================
//  DOM REFS
// ============================================================
const timerValueEl  = document.getElementById('timerValue');
const timerStateEl  = document.getElementById('timerState');
const lastRecEl     = document.getElementById('lastRecorded');
const startBtn      = document.getElementById('startBtn');
const stopBtn       = document.getElementById('stopBtn');
const logBody       = document.getElementById('logBody');
const mappingBody   = document.getElementById('mappingBody');
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');
const controlsCard  = document.getElementById('controlsCard');
const quickGrid     = document.getElementById('quickGrid');

// ============================================================
//  ESP32 COMMUNICATION
// ============================================================
async function esp32(endpoint) {
  if (!ESP32) { showToast("Enter ESP32 IP and press Connect first", "warning"); return false; }
  try {
    const controller = new AbortController();
    const timeout = setTimeout(function() { controller.abort(); }, 3000);
    await fetch(ESP32 + "/" + endpoint, { mode: "cors", signal: controller.signal });
    clearTimeout(timeout);
    setConnected(true);
    return true;
  } catch (e) {
    setConnected(false);
    return false;
  }
}

async function checkConnection() {
  if (!ESP32) return;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(function() { controller.abort(); }, 3000);
    const r = await fetch(ESP32 + "/status", { mode: "cors", signal: controller.signal });
    clearTimeout(timeout);
    setConnected(r.ok);
  } catch (e) {
    setConnected(false);
  }
}

function connectToESP32() {
  var input = document.getElementById('ipInput');
  var ip = input.value.trim();
  if (!ip) { showToast("Enter an IP address", "warning"); input.focus(); return; }

  // Clean up: remove http:// or trailing slashes if typed
  ip = ip.replace(/^https?:\/\//, '').replace(/\/+$/, '');
  input.value = ip;

  espIP = ip;
  ESP32 = "http://" + ip;
  localStorage.setItem('phase1_esp32_ip', ip);

  showToast("Connecting to " + ip + "...", "info");
  checkConnection();
}

function setConnected(val) {
  connected = val;
  statusDot.className = "status-dot " + (val ? "connected" : "disconnected");
  var connectBtn = document.getElementById('connectBtn');

  if (val) {
    statusText.textContent = "Connected (" + espIP + ")";
    connectBtn.textContent = "Connected";
    connectBtn.className = "connect-btn connected";
  } else if (espIP) {
    statusText.textContent = "Cannot reach " + espIP + " \u2014 check ESP32 is on & connected to hotspot";
    connectBtn.textContent = "Retry";
    connectBtn.className = "connect-btn";
  } else {
    statusText.textContent = "Enter the ESP32 IP address from Serial Monitor and press Connect";
    connectBtn.textContent = "Connect";
    connectBtn.className = "connect-btn";
  }
  updateButtonStates();
}

// ============================================================
//  TIMER
// ============================================================
function startTimerClock() {
  timerStartTime = performance.now();
  isRunning = true;
  tickTimer();
}

function stopTimerClock() {
  isRunning = false;
  cancelAnimationFrame(animFrame);
  return Math.round(performance.now() - timerStartTime);
}

function tickTimer() {
  if (!isRunning) return;
  const elapsed = (performance.now() - timerStartTime) / 1000;
  timerValueEl.textContent = elapsed.toFixed(3);
  animFrame = requestAnimationFrame(tickTimer);
}

// ============================================================
//  ACTIONS
// ============================================================
async function startTurn() {
  if (isRunning || !connected) return;

  playBeep(880, 0.1);
  lastRecEl.textContent = "";

  // Set speed, then start turn
  const speedOk = await esp32("speed?v=" + selectedPWM);
  if (!speedOk) { showToast("Failed to set speed — check ESP32", "warning"); return; }

  const turnOk = await esp32(selectedDir);
  if (!turnOk) { showToast("Failed to start turn — check ESP32", "warning"); return; }

  // Start timer AFTER ESP32 acknowledges the turn command
  startTimerClock();
  updateUI();
}

async function stopAndRecord() {
  if (!isRunning) return;

  // Record time BEFORE sending stop (captures user's reaction point)
  const elapsed = stopTimerClock();
  timerValueEl.textContent = (elapsed / 1000).toFixed(3);

  playBeep(440, 0.2);

  // Stop motors
  await esp32("stop");

  // Save measurement
  measurements.push({
    pwm: selectedPWM,
    dir: selectedDir,
    ms: elapsed,
    time: new Date().toLocaleTimeString()
  });
  saveMeasurements();
  renderLog();
  renderMapping();
  updateQuickGrid();
  updateUI();

  lastRecEl.textContent = "\u2713 Recorded: " + elapsed + "ms";
  showToast(selectedPWM + " " + selectedDir.toUpperCase() + " \u2192 " + elapsed + "ms", "success");
}

async function emergencyStop() {
  if (isRunning) {
    isRunning = false;
    cancelAnimationFrame(animFrame);
  }
  await esp32("stop");
  updateUI();
  showToast("\u26A0 Emergency Stop!", "warning");
}

async function quickTest(pwm, dir) {
  selectPWM(pwm);
  selectDir(dir);
  // Small delay so UI updates before starting
  await new Promise(function(r) { setTimeout(r, 50); });
  await startTurn();
}

// ============================================================
//  SELECTION
// ============================================================
function updateSliderVal() {
  document.getElementById('sliderVal').textContent = document.getElementById('pwmSlider').value;
}

function addCustomSpeed() {
  if (isRunning) return;
  var val = parseInt(document.getElementById('pwmSlider').value);
  if (!SPEEDS.includes(val)) {
    SPEEDS.push(val);
    SPEEDS.sort(function(a, b) { return a - b; });
    localStorage.setItem('phase1_speeds', JSON.stringify(SPEEDS));
    selectPWM(val);
    renderPWMButtons();
    renderMapping();
    buildQuickGrid();
    showToast("Added custom speed " + val, "success");
  } else {
    showToast("Speed " + val + " already exists", "info");
    selectPWM(val);
  }
}

function renderPWMButtons() {
  var group = document.getElementById('pwmGroup');
  group.innerHTML = "";
  SPEEDS.forEach(function(pwm) {
    var btn = document.createElement('button');
    btn.className = 'sel-btn' + (selectedPWM === pwm ? ' active' : '');
    btn.dataset.pwm = pwm;
    btn.onclick = function() { selectPWM(pwm); };
    btn.textContent = pwm;
    group.appendChild(btn);
  });
}

function selectPWM(pwm) {
  if (isRunning) return;
  selectedPWM = pwm;
  document.querySelectorAll('#pwmGroup .sel-btn').forEach(function(b) {
    b.classList.toggle('active', parseInt(b.dataset.pwm) === pwm);
  });
}

function selectDir(dir) {
  if (isRunning) return;
  selectedDir = dir;
  document.querySelectorAll('#dirGroup .sel-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.dir === dir);
  });
}

// ============================================================
//  UI UPDATES
// ============================================================
function updateUI() {
  updateButtonStates();
  timerStateEl.textContent = isRunning ? "TURNING..." : "READY";
  timerStateEl.className = "timer-state " + (isRunning ? "running" : "");
  timerValueEl.className = "timer-value " + (isRunning ? "running" : "");
  controlsCard.className = "card " + (isRunning ? "turning" : "");

  // Disable selectors during turn
  document.querySelectorAll('.sel-btn').forEach(function(b) { b.disabled = isRunning; });
  document.querySelectorAll('.quick-btn').forEach(function(b) { b.disabled = isRunning; });
}

function updateButtonStates() {
  startBtn.disabled = isRunning || !connected;
  stopBtn.disabled = !isRunning;
}

// ============================================================
//  MEASUREMENTS
// ============================================================
function saveMeasurements() {
  localStorage.setItem('phase1_cal', JSON.stringify(measurements));
}

function loadMeasurements() {
  try {
    var data = localStorage.getItem('phase1_cal');
    if (data) measurements = JSON.parse(data);
  } catch (e) { measurements = []; }
}

function renderLog() {
  logBody.innerHTML = "";
  if (measurements.length === 0) {
    logBody.innerHTML = '<tr><td colspan="5" class="empty-state">No measurements yet. Select a speed and direction, then press START.</td></tr>';
    return;
  }
  measurements.forEach(function(m, i) {
    var tr = document.createElement('tr');
    tr.className = (i === measurements.length - 1) ? "just-added" : "";
    tr.innerHTML =
      '<td style="color:#64748b">' + (i + 1) + '</td>' +
      '<td><span style="font-family:JetBrains Mono,monospace;font-weight:700">' + m.pwm + '</span></td>' +
      '<td class="dir-' + m.dir + '">' + m.dir.toUpperCase() + '</td>' +
      '<td class="time-cell">' + m.ms + 'ms</td>' +
      '<td><button class="delete-btn" onclick="deleteMeasurement(' + i + ')">&times;</button></td>';
    logBody.appendChild(tr);
  });
  // Auto-scroll to bottom
  logBody.lastElementChild.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderMapping() {
  mappingBody.innerHTML = "";
  SPEEDS.forEach(function(pwm) {
    var fwdTests   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'forward'; });
    var revTests   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'reverse'; });
    var leftTests  = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'left'; });
    var rightTests = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'right'; });
    var fwdVal   = fwdTests.length   > 0 ? fwdTests[fwdTests.length - 1].ms     : null;
    var revVal   = revTests.length   > 0 ? revTests[revTests.length - 1].ms     : null;
    var leftVal  = leftTests.length  > 0 ? leftTests[leftTests.length - 1].ms   : null;
    var rightVal = rightTests.length > 0 ? rightTests[rightTests.length - 1].ms : null;

    var tr = document.createElement('tr');
    tr.innerHTML =
      '<td style="font-family:JetBrains Mono,monospace;font-weight:700">' + pwm + '</td>' +
      '<td class="' + (fwdVal !== null ? 'has-value' : 'no-value') + '">' +
        (fwdVal !== null ? fwdVal + 'ms' : '\u2014') +
        (fwdTests.length > 1 ? '<span class="test-count">(' + fwdTests.length + ' tests)</span>' : '') +
      '</td>' +
      '<td class="' + (revVal !== null ? 'has-value' : 'no-value') + '">' +
        (revVal !== null ? revVal + 'ms' : '\u2014') +
        (revTests.length > 1 ? '<span class="test-count">(' + revTests.length + ' tests)</span>' : '') +
      '</td>' +
      '<td class="' + (leftVal !== null ? 'has-value' : 'no-value') + '">' +
        (leftVal !== null ? leftVal + 'ms' : '\u2014') +
        (leftTests.length > 1 ? '<span class="test-count">(' + leftTests.length + ' tests)</span>' : '') +
      '</td>' +
      '<td class="' + (rightVal !== null ? 'has-value' : 'no-value') + '">' +
        (rightVal !== null ? rightVal + 'ms' : '\u2014') +
        (rightTests.length > 1 ? '<span class="test-count">(' + rightTests.length + ' tests)</span>' : '') +
      '</td>';
    mappingBody.appendChild(tr);
  });
}

function deleteMeasurement(index) {
  measurements.splice(index, 1);
  saveMeasurements();
  renderLog();
  renderMapping();
  updateQuickGrid();
}

function clearAll() {
  if (measurements.length === 0) return;
  if (confirm('Clear all ' + measurements.length + ' measurements?')) {
    measurements = [];
    saveMeasurements();
    renderLog();
    renderMapping();
    updateQuickGrid();
    showToast('All measurements cleared', 'info');
  }
}

// ============================================================
//  COPY MAPPING
// ============================================================
function copyMapping() {
  var lines = [];
  lines.push('// Phase 1 Turn Calibration Results');
  lines.push('// Generated: ' + new Date().toLocaleString());
  lines.push('// Format: {pwm, fwd_ms, rev_ms, left_90deg_ms, right_90deg_ms}');
  lines.push('const int TURN_MAP[][5] = {');
  SPEEDS.forEach(function(pwm) {
    var fwd   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'forward'; });
    var rev   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'reverse'; });
    var left  = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'left'; });
    var right = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'right'; });
    var f = fwd.length   > 0 ? fwd[fwd.length - 1].ms     : 0;
    var rv = rev.length  > 0 ? rev[rev.length - 1].ms     : 0;
    var l = left.length  > 0 ? left[left.length - 1].ms   : 0;
    var r = right.length > 0 ? right[right.length - 1].ms : 0;
    lines.push('  {' + pwm + ', ' + f + ', ' + rv + ', ' + l + ', ' + r + '},');
  });
  lines.push('};');

  navigator.clipboard.writeText(lines.join('\n')).then(function() {
    showToast('Copied C++ array to clipboard!', 'success');
  });
}

function copyMappingText() {
  var lines = [];
  lines.push('Phase 1 Turn Calibration Results');
  lines.push('Generated: ' + new Date().toLocaleString());
  lines.push('');
  lines.push('PWM  | FWD (ms) | REV (ms) | LEFT 90deg (ms) | RIGHT 90deg (ms)');
  lines.push('-----|----------|----------|-----------------|------------------');
  SPEEDS.forEach(function(pwm) {
    var fwd   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'forward'; });
    var rev   = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'reverse'; });
    var left  = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'left'; });
    var right = measurements.filter(function(m) { return m.pwm === pwm && m.dir === 'right'; });
    var f = fwd.length   > 0 ? String(fwd[fwd.length - 1].ms)     : '---';
    var rv = rev.length  > 0 ? String(rev[rev.length - 1].ms)     : '---';
    var l = left.length  > 0 ? String(left[left.length - 1].ms)   : '---';
    var r = right.length > 0 ? String(right[right.length - 1].ms) : '---';
    lines.push(pwm + '  | ' + padRight(f, 8) + ' | ' + padRight(rv, 8) + ' | ' + padRight(l, 15) + ' | ' + r);
  });

  navigator.clipboard.writeText(lines.join('\n')).then(function() {
    showToast('Copied text table to clipboard!', 'success');
  });
}

function padRight(str, len) {
  while (str.length < len) str += ' ';
  return str;
}

// ============================================================
//  QUICK TEST GRID
// ============================================================
function buildQuickGrid() {
  quickGrid.innerHTML = "";
  SPEEDS.forEach(function(pwm) {
    ["forward", "reverse", "left", "right"].forEach(function(dir) {
      var btn = document.createElement('button');
      btn.className = 'quick-btn';
      btn.dataset.pwm = pwm;
      btn.dataset.dir = dir;
      btn.onclick = function() { quickTest(pwm, dir); };

      var hasData = measurements.some(function(m) { return m.pwm === pwm && m.dir === dir; });
      var arrow = '';
      if (dir === 'left') arrow = '\u2190';
      else if (dir === 'right') arrow = '\u2192';
      else if (dir === 'forward') arrow = '\u2191';
      else if (dir === 'reverse') arrow = '\u2193';

      var shortDir = dir === 'forward' ? 'FWD' : dir === 'reverse' ? 'REV' : dir.toUpperCase();

      btn.innerHTML =
        '<span class="q-pwm">' + pwm + '</span>' +
        '<span class="q-dir">' + arrow + ' ' + shortDir + '</span>' +
        (hasData ? '<span class="q-check">\u2713</span>' : '');

      quickGrid.appendChild(btn);
    });
  });
}

function updateQuickGrid() {
  buildQuickGrid();
}

// ============================================================
//  AUDIO FEEDBACK
// ============================================================
var audioCtx = null;
function playBeep(freq, duration) {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    var osc = audioCtx.createOscillator();
    var gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.frequency.value = freq;
    gain.gain.value = 0.08;
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) { /* audio not supported, ignore */ }
}

// ============================================================
//  TOAST NOTIFICATIONS
// ============================================================
function showToast(msg, type) {
  type = type || 'info';
  var toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(function() { toast.classList.add('show'); }, 10);
  setTimeout(function() {
    toast.classList.remove('show');
    setTimeout(function() { toast.remove(); }, 350);
  }, 2800);
}

// ============================================================
//  KEYBOARD SHORTCUTS
// ============================================================
document.addEventListener('keydown', function(e) {
  // SPACE to start/stop (unless typing in an input)
  if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'BUTTON') {
    e.preventDefault();
    if (isRunning) stopAndRecord();
    else startTurn();
  }
  // ESC for emergency stop
  if (e.code === 'Escape') {
    e.preventDefault();
    emergencyStop();
  }
});

// ============================================================
//  INIT
// ============================================================
var savedSpeeds = localStorage.getItem('phase1_speeds');
if (savedSpeeds) {
  try {
    SPEEDS = JSON.parse(savedSpeeds);
  } catch(e) {}
}

renderPWMButtons();
loadMeasurements();
renderLog();
renderMapping();
buildQuickGrid();

// Load saved IP from last session
var savedIP = localStorage.getItem('phase1_esp32_ip');
if (savedIP) {
  document.getElementById('ipInput').value = savedIP;
  espIP = savedIP;
  ESP32 = "http://" + savedIP;
  checkConnection();
}

// Periodically check connection (only if IP is set)
setInterval(function() { if (ESP32) checkConnection(); }, 5000);

// Enter key on IP input triggers connect
document.getElementById('ipInput').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') { e.preventDefault(); connectToESP32(); }
});

// Reset timer display
timerValueEl.textContent = '0.000';

</script>
</body>
</html>"""

# ============================================================
#  PYTHON SERVER
# ============================================================
HTML_CONTENT = HTML_TEMPLATE


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Serves the calibration dashboard HTML."""

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(HTML_CONTENT.encode('utf-8'))

    def log_message(self, format, *args):
        # Suppress default noisy logging
        pass


def main():
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Phase 1: Turn Time Mapper                     ║")
    print("  ║   Calibration Dashboard                         ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║   Dashboard : http://localhost:{DASHBOARD_PORT}               ║")
    print("  ║   ESP32 IP  : Enter on dashboard (auto-saved)   ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║   STEPS:                                        ║")
    print("  ║   1. Open Serial Monitor to find ESP32 IP       ║")
    print("  ║   2. Type the IP on the dashboard & hit Connect ║")
    print("  ║   3. Select PWM speed & direction               ║")
    print("  ║   4. Press START (or SPACEBAR) — rover turns    ║")
    print("  ║   5. Press STOP (or SPACEBAR) at 90 degrees     ║")
    print("  ║   6. Time is recorded automatically             ║")
    print("  ║   7. Repeat for all combos                      ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║   Press Ctrl+C to stop the dashboard            ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    server = http.server.HTTPServer(('localhost', DASHBOARD_PORT), DashboardHandler)

    # Auto-open browser after a short delay
    def open_browser():
        try:
            webbrowser.open(f'http://localhost:{DASHBOARD_PORT}')
        except Exception:
            pass

    threading.Timer(0.8, open_browser).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.\n")
        server.server_close()
        sys.exit(0)


if __name__ == "__main__":
    main()
