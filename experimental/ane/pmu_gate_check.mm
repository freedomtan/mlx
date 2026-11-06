// Read-only check: can this machine access ANE hardware PMU counters?
// Adapted from freedomtan/measure_ane_capacity::measure_ane_pmu.m
// checkPMUGate(). Makes no system changes.
//
// Build:
//   clang++ -fobjc-arc -std=c++17 -framework Foundation -framework Security \
//     experimental/ane/pmu_gate_check.mm -o /tmp/pmu_gate_check
// Run:
//   /tmp/pmu_gate_check

#import <Foundation/Foundation.h>
#import <Security/Security.h>
#import <sys/sysctl.h>

extern SecTaskRef SecTaskCreateFromSelf(CFAllocatorRef allocator);
extern CFTypeRef SecTaskCopyValueForEntitlement(
    SecTaskRef task, CFStringRef entitlement, CFErrorRef* error);

static BOOL has_entitlement() {
  BOOL result = NO;
  SecTaskRef task = SecTaskCreateFromSelf(NULL);
  if (task) {
    CFErrorRef err = NULL;
    CFTypeRef val = SecTaskCopyValueForEntitlement(
        task, CFSTR("com.apple.ane.hardware-counters"), &err);
    if (val) {
      if (CFGetTypeID(val) == CFBooleanGetTypeID()) {
        result = CFBooleanGetValue((CFBooleanRef)val);
      }
      CFRelease(val);
    }
    if (err) CFRelease(err);
    CFRelease(task);
  }
  return result;
}

static BOOL has_anedebug_bootarg() {
  char boot_args[1024] = {0};
  size_t size = sizeof(boot_args);
  if (sysctlbyname("kern.bootargs", boot_args, &size, NULL, 0) == 0) {
    return strstr(boot_args, "anedebug") != NULL;
  }
  return NO;
}

int main() {
  BOOL entitlement = has_entitlement();
  BOOL bootarg = has_anedebug_bootarg();
  printf("com.apple.ane.hardware-counters entitlement: %s\n",
         entitlement ? "present" : "absent");
  printf("anedebug boot-arg: %s\n", bootarg ? "present" : "absent");
  printf("PMU gate: %s\n",
         (entitlement || bootarg) ? "UNLOCKED" : "GATED");
  return 0;
}
