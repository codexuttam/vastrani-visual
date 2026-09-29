/*
 * VisionControl — Arduino Firmware (Phase 6)
 *
 * Deterministic line-based protocol hardware executor.
 * Controlled strictly by Python over USB Serial at 115200 baud.
 *
 * Protocol:
 *   COMMAND|DEVICE_ID|VALUE\n
 *
 * Examples:
 *   PING|SYSTEM
 *   ON|LIGHT_01
 *   OFF|LIGHT_01
 *   SET|LIGHT_01|80
 *   SET|FAN_01|2
 *   SET|SERVO_01|90
 *   ALL_OFF|SYSTEM
 *
 * Pin Mapping:
 *   LIGHT_PIN : Pin 9  (PWM capable)
 *   FAN_PIN   : Pin 8  (Safe low-voltage relay / LED indicator)
 *   SERVO_PIN : Pin 10 (PWM Servo output)
 *   MUSIC_PIN : Pin 11 (LED indicator)
 */

#include <Servo.h>

// ─── Pin Configuration ──────────────────────────────────────────────────────
const int LIGHT_PIN = 9;   // PWM capable pin for brightness control (0-255)
const int FAN_PIN   = 8;   // Digital pin for fan speed / relay output
const int SERVO_PIN = 10;  // Servo control pin
const int MUSIC_PIN = 11;  // Music status LED indicator

// ─── Hardware Objects & State ────────────────────────────────────────────────
Servo servoDevice;

bool lightPower = false;
int lightLevel  = 0;

bool fanPower = false;
int fanLevel  = 0;

bool musicPower = false;

int servoAngle = 0;

String inputBuffer = "";

// ─── Setup ───────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);

  // Initialize output pins
  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  pinMode(MUSIC_PIN, OUTPUT);

  servoDevice.attach(SERVO_PIN);

  // Requirement 22: Startup safe state - ALL OUTPUTS OFF
  allOff();

  inputBuffer.reserve(64);
}

// ─── Main Loop ───────────────────────────────────────────────────────────────

void loop() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    if (inChar == '\n' || inChar == '\r') {
      if (inputBuffer.length() > 0) {
        processCommand(inputBuffer);
        inputBuffer = "";
      }
    } else {
      inputBuffer += inChar;
    }
  }
}

// ─── Command Processor ───────────────────────────────────────────────────────

void processCommand(String cmdLine) {
  cmdLine.trim();
  if (cmdLine.length() == 0) return;

  // Split by '|'
  int idx1 = cmdLine.indexOf('|');
  int idx2 = cmdLine.indexOf('|', idx1 + 1);

  String action = "";
  String deviceId = "";
  String valStr = "";

  if (idx1 == -1) {
    action = cmdLine;
  } else if (idx2 == -1) {
    action = cmdLine.substring(0, idx1);
    deviceId = cmdLine.substring(idx1 + 1);
  } else {
    action = cmdLine.substring(0, idx1);
    deviceId = cmdLine.substring(idx1 + 1, idx2);
    valStr = cmdLine.substring(idx2 + 1);
  }

  action.toUpperCase();
  deviceId.toUpperCase();

  // Handshake & System commands
  if (action == "PING") {
    Serial.println("PONG|SYSTEM");
    return;
  }

  if (action == "ALL_OFF" || action == "EMERGENCY_STOP") {
    allOff();
    Serial.println("OK|SYSTEM|ALL_OFF");
    return;
  }

  // LIGHT_01
  if (deviceId == "LIGHT_01") {
    if (action == "ON") {
      lightPower = true;
      lightLevel = 100;
      analogWrite(LIGHT_PIN, 255);
      Serial.println("OK|LIGHT_01|ON");
    } else if (action == "OFF") {
      lightPower = false;
      lightLevel = 0;
      analogWrite(LIGHT_PIN, 0);
      Serial.println("OK|LIGHT_01|OFF");
    } else if (action == "TOGGLE") {
      lightPower = !lightPower;
      lightLevel = lightPower ? 100 : 0;
      analogWrite(LIGHT_PIN, lightPower ? 255 : 0);
      Serial.println("OK|LIGHT_01|" + String(lightPower ? "ON" : "OFF"));
    } else if (action == "SET") {
      int val = valStr.toInt();
      if (val < 0 || val > 100) {
        Serial.println("ERR|INVALID_VALUE");
      } else {
        lightLevel = val;
        lightPower = (val > 0);
        int pwmVal = map(val, 0, 100, 0, 255);
        analogWrite(LIGHT_PIN, pwmVal);
        Serial.println("OK|LIGHT_01|" + String(val));
      }
    } else {
      Serial.println("ERR|INVALID_COMMAND");
    }
    return;
  }

  // FAN_01
  if (deviceId == "FAN_01") {
    if (action == "ON") {
      fanPower = true;
      fanLevel = 1;
      digitalWrite(FAN_PIN, HIGH);
      Serial.println("OK|FAN_01|ON");
    } else if (action == "OFF") {
      fanPower = false;
      fanLevel = 0;
      digitalWrite(FAN_PIN, LOW);
      Serial.println("OK|FAN_01|OFF");
    } else if (action == "TOGGLE") {
      fanPower = !fanPower;
      fanLevel = fanPower ? 1 : 0;
      digitalWrite(FAN_PIN, fanPower ? HIGH : LOW);
      Serial.println("OK|FAN_01|" + String(fanPower ? "ON" : "OFF"));
    } else if (action == "SET") {
      int val = valStr.toInt();
      if (val < 0 || val > 3) {
        Serial.println("ERR|INVALID_VALUE");
      } else {
        fanLevel = val;
        fanPower = (val > 0);
        digitalWrite(FAN_PIN, fanPower ? HIGH : LOW);
        Serial.println("OK|FAN_01|" + String(val));
      }
    } else {
      Serial.println("ERR|INVALID_COMMAND");
    }
    return;
  }

  // MUSIC_01 (Simulated output LED indicator)
  if (deviceId == "MUSIC_01") {
    if (action == "ON") {
      musicPower = true;
      digitalWrite(MUSIC_PIN, HIGH);
      Serial.println("OK|MUSIC_01|ON");
    } else if (action == "OFF") {
      musicPower = false;
      digitalWrite(MUSIC_PIN, LOW);
      Serial.println("OK|MUSIC_01|OFF");
    } else if (action == "TOGGLE") {
      musicPower = !musicPower;
      digitalWrite(MUSIC_PIN, musicPower ? HIGH : LOW);
      Serial.println("OK|MUSIC_01|" + String(musicPower ? "ON" : "OFF"));
    } else if (action == "SET") {
      int val = valStr.toInt();
      if (val < 0 || val > 100) {
        Serial.println("ERR|INVALID_VALUE");
      } else {
        musicPower = (val > 0);
        digitalWrite(MUSIC_PIN, musicPower ? HIGH : LOW);
        Serial.println("OK|MUSIC_01|" + String(val));
      }
    } else {
      Serial.println("ERR|INVALID_COMMAND");
    }
    return;
  }

  // SERVO_01
  if (deviceId == "SERVO_01") {
    if (action == "SET") {
      int val = valStr.toInt();
      if (val < 0 || val > 180) {
        Serial.println("ERR|INVALID_VALUE");
      } else {
        servoAngle = val;
        servoDevice.write(servoAngle);
        Serial.println("OK|SERVO_01|" + String(val));
      }
    } else if (action == "ON" || action == "OFF") {
      servoAngle = (action == "ON") ? 90 : 0;
      servoDevice.write(servoAngle);
      Serial.println("OK|SERVO_01|" + String(servoAngle));
    } else {
      Serial.println("ERR|INVALID_COMMAND");
    }
    return;
  }

  // Unknown device ID
  Serial.println("ERR|UNKNOWN_DEVICE");
}

// ─── Emergency Stop Helper ───────────────────────────────────────────────────

void allOff() {
  lightPower = false;
  lightLevel = 0;
  analogWrite(LIGHT_PIN, 0);

  fanPower = false;
  fanLevel = 0;
  digitalWrite(FAN_PIN, LOW);

  musicPower = false;
  digitalWrite(MUSIC_PIN, LOW);

  servoAngle = 0;
  servoDevice.write(0);
}
