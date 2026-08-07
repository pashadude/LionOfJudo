/*
 * lionimu — wearable IMU logger for judo throw analysis.
 *
 * Board: ESP32-S3 SuperMini (16MB flash, N16R8). Arduino core for ESP32.
 * Sensor: MPU-6050 (GY-521) over I2C at 400kHz.
 *
 * Boot behaviour:
 *   - Scan for HOME_SSID for WIFI_SCAN_TIMEOUT_MS.
 *   - Found (you're at home):   join, mDNS as lion-<unit>.local, serve HTTP
 *                               /list /download?f= /delete?f= /status.
 *                               Slow green blink.
 *   - Not found (at the dojo):  10s countdown blink, then log to LittleFS
 *                               at SAMPLE_RATE_HZ until power-off/low batt.
 *                               Fast blink on >HIGHG_BLINK_G events.
 *
 * Log format (little-endian) — must match pipeline/imu_ingest.py:
 *   Header 32B:  'LJIM' | u8 version=1 | u8 unit_id | u16 rate |
 *                f32 accel_scale | f32 gyro_scale | 16B reserved
 *   Record 16B:  u32 millis | i16 ax,ay,az | i16 gx,gy,gz
 *
 * Arduino IDE settings: Board "ESP32S3 Dev Module", Flash Size 16MB,
 * Partition Scheme "16M Flash (3MB APP/9.9MB FATFS)" -> use "Custom" with
 * LittleFS, or simply "8M with spiffs" variants that give LittleFS >= 9MB.
 * Libraries: none beyond the ESP32 core (WiFi, WebServer, ESPmDNS, LittleFS,
 * Wire) + Adafruit NeoPixel for the onboard RGB LED.
 */

#include <Adafruit_NeoPixel.h>
#include <ESPmDNS.h>
#include <LittleFS.h>
#include <WebServer.h>
#include <WiFi.h>
#include <Wire.h>

#include "config.h"
#include "battery_policy.h"

// ---- MPU-6050 registers (same map as pico_unified_controller.py) ----
#define MPU_ADDR        0x68
#define REG_PWR_MGMT_1  0x6B
#define REG_SMPLRT_DIV  0x19
#define REG_CONFIG      0x1A
#define REG_GYRO_CFG    0x1B
#define REG_ACCEL_CFG   0x1C
#define REG_ACCEL_XOUT  0x3B

WebServer server(80);
Adafruit_NeoPixel led(1, PIN_LED, NEO_GRB + NEO_KHZ800);

File logFile;
uint8_t writeBuf[FLUSH_BLOCK];
size_t bufPos = 0;
bool logging = false;
uint32_t sampleCount = 0;
uint32_t nextSampleUs = 0;
const uint32_t sampleIntervalUs = 1000000UL / SAMPLE_RATE_HZ;

// ---------------------------------------------------------------- MPU-6050

static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static bool mpuInit() {
  Wire.begin(PIN_SDA, PIN_SCL, 400000);
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) return false;

  mpuWrite(REG_PWR_MGMT_1, 0x01);  // wake, PLL gyro-X clock
  delay(100);
  mpuWrite(REG_CONFIG, 0x02);      // DLPF ~94Hz
  mpuWrite(REG_SMPLRT_DIV, 1000 / SAMPLE_RATE_HZ - 1);
  mpuWrite(REG_GYRO_CFG, 0x18);    // +/-2000 dps
  mpuWrite(REG_ACCEL_CFG, 0x18);   // +/-16 g
  return true;
}

// Reads 14 bytes: accel(6) + temp(2) + gyro(6)
static bool mpuRead(int16_t a[3], int16_t g[3]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, 14) != 14) return false;

  uint8_t raw[14];
  for (int i = 0; i < 14; i++) raw[i] = Wire.read();
  a[0] = (raw[0] << 8) | raw[1];
  a[1] = (raw[2] << 8) | raw[3];
  a[2] = (raw[4] << 8) | raw[5];
  g[0] = (raw[8] << 8) | raw[9];
  g[1] = (raw[10] << 8) | raw[11];
  g[2] = (raw[12] << 8) | raw[13];
  return true;
}

// ---------------------------------------------------------------- LED

static void ledColor(uint8_t r, uint8_t g, uint8_t b) {
  led.setPixelColor(0, led.Color(r, g, b));
  led.show();
}

// ---------------------------------------------------------------- battery

static float batteryVolts() {
  // The 100k/100k divider is intentionally low-drain.  Discard one ADC read
  // and average the next four; add a 100nF capacitor from GPIO4 to GND in the
  // physical build so this high-impedance source is stable.
  analogReadMilliVolts(PIN_BATTERY_ADC);
  uint32_t totalMv = 0;
  for (int i = 0; i < 4; i++) totalMv += analogReadMilliVolts(PIN_BATTERY_ADC);
  return (totalMv / 4.0f) / 1000.0f * BATT_DIVIDER;
}

// ---------------------------------------------------------------- logging

static void writeHeader(File &f) {
  uint8_t h[32] = {0};
  memcpy(h, "LJIM", 4);
  h[4] = 1;                       // version
  h[5] = UNIT_ID;
  uint16_t rate = SAMPLE_RATE_HZ;
  memcpy(h + 6, &rate, 2);
  float as = ACCEL_SCALE, gs = GYRO_SCALE;
  memcpy(h + 8, &as, 4);
  memcpy(h + 12, &gs, 4);
  f.write(h, 32);
}

static String nextLogName() {
  int maxN = 0;
  File dir = LittleFS.open(LOG_DIR);
  File f;
  while ((f = dir.openNextFile())) {
    String n = f.name();          // e.g. "003.bin"
    int v = n.toInt();
    if (v > maxN) maxN = v;
    f.close();
  }
  char buf[16];
  snprintf(buf, sizeof(buf), LOG_DIR "/%03d.bin", maxN + 1);
  return String(buf);
}

static void startLogging() {
  LittleFS.mkdir(LOG_DIR);
  String name = nextLogName();
  logFile = LittleFS.open(name, FILE_WRITE);
  if (!logFile) {
    // Flash full or FS broken: solid red, halt.
    ledColor(255, 0, 0);
    while (true) delay(1000);
  }
  writeHeader(logFile);
  logging = true;
  nextSampleUs = micros();
  Serial.printf("logging to %s\n", name.c_str());
}

static void stopLogging() {
  if (!logging) return;
  if (bufPos > 0) logFile.write(writeBuf, bufPos);
  logFile.close();
  logging = false;
  Serial.printf("closed log, %u samples\n", sampleCount);
}

static void logSample() {
  // Paced by micros() so WiFi-off logging holds SAMPLE_RATE_HZ steadily.
  uint32_t now = micros();
  if ((int32_t)(now - nextSampleUs) < 0) return;
  nextSampleUs += sampleIntervalUs;

  int16_t a[3], g[3];
  if (!mpuRead(a, g)) return;

  uint32_t ms = millis();
  uint8_t rec[16];
  memcpy(rec, &ms, 4);
  memcpy(rec + 4, a, 6);
  memcpy(rec + 10, g, 6);

  memcpy(writeBuf + bufPos, rec, 16);
  bufPos += 16;
  if (bufPos >= FLUSH_BLOCK) {
    logFile.write(writeBuf, bufPos);
    bufPos = 0;
  }
  sampleCount++;

  // liveness blink on impacts
  float totalG = sqrtf((float)a[0] * a[0] + (float)a[1] * a[1] +
                       (float)a[2] * a[2]) / ACCEL_SCALE;
  if (totalG > HIGHG_BLINK_G) ledColor(0, 0, 120);
  else if ((sampleCount & 0x3FF) == 0) ledColor(0, 25, 0);  // dim heartbeat
  else if ((sampleCount & 0x3FF) == 64) ledColor(0, 0, 0);

  // battery check every ~5s
  if (sampleCount % (SAMPLE_RATE_HZ * 5) == 0) {
    if (shouldStopForLowBattery(batteryVolts(), BATT_LOW_V,
                                !BENCH_USB_POWER)) {
      stopLogging();
      while (true) {              // red blink until power-off
        ledColor(255, 0, 0); delay(300);
        ledColor(0, 0, 0);   delay(700);
      }
    }
  }
}

// ---------------------------------------------------------------- HTTP

static void handleList() {
  String out = "[";
  File dir = LittleFS.open(LOG_DIR);
  File f;
  bool first = true;
  while ((f = dir.openNextFile())) {
    if (!first) out += ",";
    out += "{\"name\":\"" + String(f.name()) + "\",\"size\":" +
           String(f.size()) + "}";
    first = false;
    f.close();
  }
  out += "]";
  server.send(200, "application/json", out);
}

static void handleDownload() {
  String name = server.arg("f");
  File f = LittleFS.open(String(LOG_DIR) + "/" + name, FILE_READ);
  if (!f) { server.send(404, "text/plain", "no such file"); return; }
  server.streamFile(f, "application/octet-stream");
  f.close();
}

static void handleDelete() {
  String name = server.arg("f");
  bool ok = LittleFS.remove(String(LOG_DIR) + "/" + name);
  server.send(ok ? 200 : 404, "text/plain", ok ? "deleted" : "not found");
}

static void handleStatus() {
  String out = "{\"unit\":\"" UNIT_NAME "\",\"battery_v\":" +
               String(batteryVolts(), 2) + ",\"free_kb\":" +
               String((LittleFS.totalBytes() - LittleFS.usedBytes()) / 1024) +
               "}";
  server.send(200, "application/json", out);
}

// ---------------------------------------------------------------- modes

static bool tryHomeWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(HOME_SSID, HOME_PASS);
  uint32_t t0 = millis();
  while (millis() - t0 < WIFI_SCAN_TIMEOUT_MS) {
    if (WiFi.status() == WL_CONNECTED) return true;
    delay(200);
  }
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  return false;
}

static void downloadMode() {
  MDNS.begin("lion-" UNIT_NAME);
  server.on("/list", handleList);
  server.on("/download", handleDownload);
  server.on("/delete", HTTP_POST, handleDelete);
  server.on("/status", handleStatus);
  server.begin();
  Serial.printf("download mode: http://lion-%s.local\n", UNIT_NAME);

  uint32_t lastBlink = 0;
  bool on = false;
  while (true) {
    server.handleClient();
    if (millis() - lastBlink > 1000) {   // slow green blink
      on = !on;
      ledColor(0, on ? 60 : 0, 0);
      lastBlink = millis();
    }
    delay(2);
  }
}

static void loggingMode() {
  // 10s countdown so you can still power-cycle before a session starts
  for (int i = 10; i > 0; i--) {
    ledColor(60, 40, 0); delay(500);
    ledColor(0, 0, 0);   delay(500);
  }
  startLogging();
}

// ---------------------------------------------------------------- main

void setup() {
  Serial.begin(115200);
#if BENCH_USB_POWER
  Serial.println("BENCH USB POWER: battery cutoff disabled");
#endif
  led.begin();
  ledColor(20, 20, 20);
  analogSetPinAttenuation(PIN_BATTERY_ADC, ADC_11db);

  // Partition scheme app3M_fat9M_16MB labels its 9.9MB data partition
  // "ffat"; LittleFS happily formats/mounts it under that label.
  if (!LittleFS.begin(true, "/littlefs", 10, "ffat")) {
    ledColor(255, 0, 0);
    while (true) delay(1000);
  }
  if (!mpuInit()) {
    // sensor missing: purple blink forever
    while (true) {
      ledColor(120, 0, 120); delay(300);
      ledColor(0, 0, 0);     delay(300);
    }
  }

  if (tryHomeWifi()) downloadMode();   // never returns
  loggingMode();
}

void loop() {
  if (logging) logSample();
}
