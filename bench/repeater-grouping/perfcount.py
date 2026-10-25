"""Counts user-space instructions retired by a command and its children via perf_event_open (ctypes).

Linux x86_64 only. Needs /proc/sys/kernel/perf_event_paranoid <= 2 (user space is counted, not the kernel).
On a hybrid CPU (Intel P/E cores) the event is opened on the cpu_core PMU, so pin the process to the P-cores
with BENCH_CPUS (see pin_from_env). Without the counter the benchmark still works: `ginstr` is null.
"""
import ctypes, os, struct, subprocess, fcntl


def pin_from_env():
    """Pins this process (and so every child) to the CPUs of $BENCH_CPUS, e.g. "0-11" or "0-3,6". Unset: no pinning."""
    spec = os.environ.get("BENCH_CPUS", "").strip()
    if not spec or not hasattr(os, "sched_setaffinity"):
        return
    cpus = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    os.sched_setaffinity(0, cpus)


def pmu_type():
    """The perf PMU type to put in the high bits of the config: cpu_core on a hybrid CPU, else 0 (generic)."""
    try:
        return int(open("/sys/devices/cpu_core/type").read())
    except (OSError, ValueError):
        return 0

PERF_TYPE_HARDWARE = 0
PERF_COUNT_HW_INSTRUCTIONS = 1
PERF_COUNT_HW_CPU_CYCLES = 0
SYS_perf_event_open = 298  # x86_64
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_DISABLE = 0x2401
PERF_EVENT_IOC_RESET = 0x2403

class Counter:
    def __init__(self, config=PERF_COUNT_HW_INSTRUCTIONS, pmu=None):
        pmu = pmu_type() if pmu is None else pmu
        # struct perf_event_attr (size 128): type u32, size u32, config u64, sample_period u64, sample_type u64,
        # read_format u64, flags u64 (bit0 disabled, bit1 inherit, bit5 exclude_kernel, bit6 exclude_hv), ...
        attr = bytearray(128)
        struct.pack_into("IIQQQQQ", attr, 0, PERF_TYPE_HARDWARE, 128, config | (pmu << 32), 0, 0, 0,
                         (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6))
        libc = ctypes.CDLL(None, use_errno=True)
        buf = (ctypes.c_char * 128).from_buffer(attr)
        fd = libc.syscall(SYS_perf_event_open, ctypes.byref(buf), 0, -1, -1, 0)
        if fd < 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        self.fd = fd

    def enable(self):
        fcntl.ioctl(self.fd, PERF_EVENT_IOC_RESET, 0)
        fcntl.ioctl(self.fd, PERF_EVENT_IOC_ENABLE, 0)

    def disable(self):
        fcntl.ioctl(self.fd, PERF_EVENT_IOC_DISABLE, 0)

    def read(self):
        return struct.unpack("Q", os.read(self.fd, 8))[0]

if __name__ == "__main__":
    import sys
    pin_from_env()
    c = Counter()
    c.enable()
    subprocess.run(sys.argv[1:])
    c.disable()
    print("instructions (user):", c.read())
