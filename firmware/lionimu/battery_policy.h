#pragma once

static inline bool shouldStopForLowBattery(float voltage, float threshold,
                                           bool monitorEnabled) {
  return monitorEnabled && voltage < threshold;
}
