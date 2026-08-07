#include <cassert>

#include "../firmware/lionimu/battery_policy.h"

int main() {
  assert(shouldStopForLowBattery(0.0f, 3.30f, true));
  assert(!shouldStopForLowBattery(3.70f, 3.30f, true));
  assert(!shouldStopForLowBattery(0.0f, 3.30f, false));
  return 0;
}
