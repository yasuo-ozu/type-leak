#!/usr/bin/env python3
"""measure.py: clean build/check timing of one generated workspace.

  measure.py DIR [--tc stable] -> prints a JSON object with cpu/wall (s) of
  build_leaker, build_referrer, check_leaker, check_referrer, and the size in bytes of the
  artifact (`size`: the .rlib after a build, the .rmeta after a check).
  measure.py DIR --sizes        -> only the sizes (and a sha1 of each artifact), no timing.
"""
import glob, hashlib, json, os, resource, shutil, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from perfcount import Counter, pin_from_env
except Exception:  # pragma: no cover
    Counter = None
    pin_from_env = lambda: None

# On a hybrid CPU (P-cores and E-cores) a process that migrates between the core types takes up to twice
# the CPU time, and the instruction counter (cpu_core PMU) counts the P-cores only. Set BENCH_CPUS to the
# P-core threads (e.g. BENCH_CPUS=0-11, see lscpu -e); every child inherits the pinning.
try:
    pin_from_env()
except Exception:
    pass

def run(cmd, cwd, env):
    ctr = None
    try:
        ctr = Counter() if Counter else None
    except Exception:
        ctr = None
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    if ctr:
        ctr.enable()
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    instr = None
    if ctr:
        ctr.disable()
        instr = ctr.read()
        os.close(ctr.fd)
    wall = time.monotonic() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    user = r1.ru_utime - r0.ru_utime
    sys_ = r1.ru_stime - r0.ru_stime
    return dict(cpu=round(user + sys_, 3), user=round(user, 3), sys=round(sys_, 3), wall=round(wall, 3),
                rc=p.returncode, ginstr=(round(instr / 1e9, 4) if instr is not None else None), err=(p.stdout[-400:] if p.returncode else ""))

def artifact(d, crate, phase):
    """Path of the crate's .rlib (after a build) or .rmeta (after a check), or None unless there is exactly one."""
    ext = "rlib" if phase == "build" else "rmeta"
    found = glob.glob(os.path.join(d, "target", "debug", "deps", "lib%s-*.%s" % (crate, ext)))
    return found[0] if len(found) == 1 else None


def artifact_size(d, crate, phase):
    p = artifact(d, crate, phase)
    return os.path.getsize(p) if p else None


def cargo_env(env_extra=None):
    env = dict(os.environ)
    env["CARGO_INCREMENTAL"] = env.get("CARGO_INCREMENTAL", "0")
    env["CARGO_TERM_COLOR"] = "never"
    env["CARGO_TERM_PROGRESS_WHEN"] = "never"
    if env_extra:
        env.update(env_extra)
    return env


def measure(d, tc="stable", phases=("build", "check"), env_extra=None):
    env = cargo_env(env_extra)
    out = dict(load=open("/proc/loadavg").read().split()[:3])
    try:  # an old cargo (1.56) cannot read the lockfile format of a newer one
        os.remove(os.path.join(d, "Cargo.lock"))
    except OSError:
        pass
    for ph in phases:
        shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
        for crate in ("leaker", "referrer"):
            cmd = ["cargo", "+" + tc, ph, "--offline", "-q", "-p", crate]
            res = run(cmd, d, env)
            res["size"] = artifact_size(d, crate, ph)  # before the target dir is removed
            out["%s_%s" % (ph, crate)] = res
    shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
    return out


def measure_sizes(d, tc="stable", env_extra=None, remap=True):
    """Builds and checks both crates once, without timing: {phase_crate: {rc, size, sha1}}.

    The artifacts contain the absolute source paths, so their size depends on the length of the path of the
    workspace by a few bytes. With `remap` the workspace is mapped to /ws, and the bytes are the same wherever
    the benchmark is run (same rustc)."""
    env = cargo_env(env_extra)
    if remap:
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "") + " --remap-path-prefix=%s=/ws" % os.path.abspath(d)).strip()
    out = {}
    try:
        os.remove(os.path.join(d, "Cargo.lock"))
    except OSError:
        pass
    for ph in ("build", "check"):
        shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
        for crate in ("leaker", "referrer"):
            p = subprocess.run(["cargo", "+" + tc, ph, "--offline", "-q", "-p", crate], cwd=d, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            path = artifact(d, crate, ph)
            out["%s_%s" % (ph, crate)] = dict(
                rc=p.returncode, size=os.path.getsize(path) if path else None,
                sha1=hashlib.sha1(open(path, "rb").read()).hexdigest()[:16] if path else None,
                err=(p.stdout[-400:] if p.returncode else ""))
    shutil.rmtree(os.path.join(d, "target"), ignore_errors=True)
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tc", default="stable")
    ap.add_argument("--sizes", action="store_true", help="only the artifact sizes, no timing")
    ap.add_argument("--no-remap", action="store_true", help="with --sizes: do not map the workspace path to /ws")
    a = ap.parse_args()
    print(json.dumps(measure_sizes(a.dir, a.tc, remap=not a.no_remap) if a.sizes else measure(a.dir, a.tc), indent=1))
