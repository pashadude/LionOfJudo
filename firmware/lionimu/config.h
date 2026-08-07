#pragma once

// ---- Per-unit identity (change before flashing each unit!) ----
#define UNIT_ID        0            // 0 = chest, 1 = hip
#define UNIT_NAME      "chest"      // "chest" -> lion-chest.local

// ---- WiFi (download mode) ----
// Keep real credentials in the git-ignored secrets.h. A clean checkout still
// compiles with placeholders, but will enter logging mode because that network
// is not expected to exist.
#if __has_include("secrets.h")
#include "secrets.h"
#endif
#ifndef HOME_SSID
#define HOME_SSID      "YOUR_HOME_WIFI"
#endif
#ifndef HOME_PASS
#define HOME_PASS      "YOUR_HOME_PASSWORD"
#endif
#define WIFI_SCAN_TIMEOUT_MS  8000

// ---- Sampling ----
#define SAMPLE_RATE_HZ   200
#define ACCEL_RANGE_16G  1          // +/-16g (judo impacts)
#define GYRO_RANGE_2000  1          // +/-2000 deg/s
// Scale factors matching the ranges above (LSB per unit)
#define ACCEL_SCALE      2048.0f    // LSB/g at +/-16g
#define GYRO_SCALE       16.4f      // LSB/(deg/s) at +/-2000dps

// ---- Pins (ESP32-S3 SuperMini) ----
#define PIN_SDA          8
#define PIN_SCL          9
#define PIN_LED          48         // onboard RGB LED data pin (WS2812)
#define PIN_BATTERY_ADC  4          // through 100k/100k divider to cell +

// ---- Battery ----
#define BATT_DIVIDER     2.0f       // 100k/100k
#define BATT_LOW_V       3.30f      // close log + red blink below this
#ifndef BENCH_USB_POWER
#define BENCH_USB_POWER  0          // 1 only for tethered/power-bank bench use
#endif

// ---- Logging ----
#define LOG_DIR          "/logs"
#define FLUSH_BLOCK      4096       // bytes per LittleFS write
#define HIGHG_BLINK_G    3.0f       // fast blink above this (visual liveness)
