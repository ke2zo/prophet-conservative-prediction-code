"""Re-run the computations behind the manuscript and compare them with the saved outputs.

Usage (from this folder, journal/checks):
  python3 reproduce.py                  tier 'quick': self-tests, all checks and certificates that take minutes
  python3 reproduce.py --tier full      also the long runs: the four certification runs of Section 5.2 (from a
                                        fresh float curve), the gradient check against 30 digits, and the
                                        horizons n = 7, 8 of Proposition 7.8 (hours; see README.md)
  python3 reproduce.py --tier search    also the local searches behind the evidence of Section 7 (optional)
  python3 reproduce.py --only NAME ...  selected steps (see --list)
  python3 reproduce.py --only arb_check the cross-check of the certification with Arb (needs python-flint; about
                                        45 minutes with 4 processes); it belongs to no tier
Options: --procs P (processes for the parallel runs, default 4), --workdir DIR (default ./_reproduce).

The folder is first copied to the work directory, so the saved outputs are never overwritten. Each step runs
there, and every output it writes is compared with the saved one after removing run times. A step PASSES if it
exits normally and its outputs agree; DIFF lists the first differing lines. Outputs of the searches, and the
json files of the certification runs, are compared for information only ('info'). A report is written to
WORKDIR/reproduce_report.txt.
"""
import argparse, gzip, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# (name, tier, folder, [commands], outputs, compare, env)
#   a command is (argv, stdout target or None, mode 'w' or 'a');  compare: 'exact' or 'info'
def steps(P):
    ext = {'CURVE': 'curve_float_ext.npy', 'PROCS': str(P)}
    S = []
    def add(name, tier, folder, cmds, outs, compare='exact', env=None):
        S.append(dict(name=name, tier=tier, folder=folder, cmds=cmds, outs=outs, compare=compare, env=env or {}))
    # ---- Section 5.2: interval certification (certify/)
    add('ivec_selftest', 'quick', 'certify', [(['ivec.py'], None, 'w')], [])
    add('curve_float', 'full', 'certify', [(['curve_float.py'], None, 'w')], ['curve_float.npy'], 'info')
    add('curve_float_ext', 'full', 'certify', [(['curve_float_ext.py'], None, 'w')], ['curve_float_ext.npy'], 'info')
    add('cert_1.001_1000', 'full', 'certify',
        [(['-u', 'run_certify.py', '1.001', '1000', '1.003', '1.01', '1.05', '1.25', '2', '5', '20', '50', '100', '300'],
          'cert_1.001_1000.log', 'w')], ['cert_1.001_1000.log', 'cert_1.001_1000.json.gz'], 'exact', {'PROCS': str(P)})
    add('cert_1.0001_1.001', 'full', 'certify', [(['-u', 'run_certify.py', '1.0001', '1.001', '1.0003'],
        'cert_1.0001_1.001.log', 'w')], ['cert_1.0001_1.001.log', 'cert_1.0001_1.001.json.gz'], 'exact', ext)
    add('cert_1000_3000', 'full', 'certify', [(['-u', 'run_certify.py', '1000', '3000', '1500', '2200'],
        'cert_1000_3000.log', 'w')], ['cert_1000_3000.log', 'cert_1000_3000.json.gz'], 'exact', ext)
    add('cert_2200_3000', 'full', 'certify', [(['-u', 'run_certify.py', '2200', '3000'], 'cert_2200_3000.log', 'w')],
        ['cert_2200_3000.log', 'cert_2200_3000.json.gz'], 'exact',
        dict(ext, TUBE='tube_0.721383_0.781416_1e-07.json', MAXLEVEL='20'))
    add('glue_tubes', 'quick', 'certify',
        [(['glue_tubes.py', 'cert_1.001_1000.json.gz', 'cert_1.0001_1.001.json.gz'], 'glue_tubes.txt', 'w'),
         (['glue_tubes.py', 'cert_1.001_1000.json.gz', 'cert_1000_3000.json.gz'], 'glue_tubes.txt', 'a')],
        ['glue_tubes.txt'])
    add('point_values', 'quick', 'certify', [(['point_values.py'], 'point_values.txt', 'w')], ['point_values.txt'])
    add('test_boxes', 'quick', 'certify', [(['test_boxes.py', '2026'], 'test_boxes.txt', 'w')], ['test_boxes.txt'])
    add('test_grads', 'quick', 'certify', [(['test_grads.py'], 'test_grads.txt', 'w')], ['test_grads.txt'])
    add('test_grads_mp', 'full', 'certify', [(['test_grads_mp.py', '100', '2026'], 'test_grads_mp.txt', 'w')],
        ['test_grads_mp.txt'])
    # cross-check with Arb; needs python-flint, so it is in no tier: run it with --only arb_check
    add('arb_check', 'arb', 'certify', [(['-u', 'arb_check.py'], 'arb_check.txt', 'w')], ['arb_check.txt'], 'exact',
        {'PROCS': str(P)})
    # ---- Section 4: sanity checks (top level; not used in any proof)
    add('firstvar_atom_check', 'quick', '.', [(['firstvar_atom_check.py'], 'firstvar_atom_check.txt', 'w')],
        ['firstvar_atom_check.txt'])
    add('free_minimization_K80', 'quick', '.', [(['free_minimization_check.py'], 'free_minimization_K80.txt', 'w')],
        ['free_minimization_K80.txt'], 'exact', {'FREEMIN_K': '80', 'FREEMIN_C': '2,5,10'})
    add('free_minimization_K160', 'quick', '.', [(['free_minimization_check.py'], 'free_minimization_K160.txt', 'w')],
        ['free_minimization_K160.txt'], 'exact', {'FREEMIN_K': '160', 'FREEMIN_C': '2'})
    add('d1_scan', 'quick', '.', [(['d1_scan.py'], 'd1_scan.txt', 'w')], ['d1_scan.txt'])
    add('d1_no_critical_points', 'quick', '.', [(['d1_no_critical_points.py'], 'd1_no_critical_points.txt', 'w')],
        ['d1_no_critical_points.txt'])
    # ---- uniqueness evidence (Supplement (e))
    add('reduce2d', 'quick', 'uniqueness', [(['reduce2d.py'], 'reduce2d.txt', 'w')], ['reduce2d.txt'])
    add('gamma_grid', 'quick', 'uniqueness', [(['gamma_grid.py'], 'gamma_grid.txt', 'w')], ['gamma_grid.txt'])
    add('dgamma', 'quick', 'uniqueness', [(['dgamma.py'], 'dgamma.txt', 'w')], ['dgamma.txt'])
    add('colcheck', 'quick', 'uniqueness', [(['colcheck.py'], 'colcheck.txt', 'w')], ['colcheck.txt'])
    add('contour', 'quick', 'uniqueness', [(['contour.py'], 'contour.txt', 'w')], ['contour.txt'])
    add('boundary_roots', 'quick', 'uniqueness', [(['boundary_roots.py'], None, 'w')], ['boundary_roots.txt'])
    add('runmap', 'quick', 'uniqueness', [(['runmap.py'], 'runmap.txt', 'w')], ['runmap.txt'])
    # ---- Sections 6.1-6.2 and Lemma 5.3 (asymptotics/)
    add('closed_form_check', 'quick', 'asymptotics', [(['closed_form_check.py'], 'closed_form_check.txt', 'w')],
        ['closed_form_check.txt'])
    add('corner_limits', 'quick', 'asymptotics', [(['corner_limits.py'], 'corner_limits.txt', 'w')],
        ['corner_limits.txt'])
    add('indep_identity_check', 'quick', 'asymptotics', [(['indep_identity_check.py'], 'indep_identity_check.txt', 'w')],
        ['indep_identity_check.txt'])
    add('indep_corner', 'quick', 'asymptotics',
        [(['indep_corner.py', '0.01', '0.001', '0.0001', '1.0e-5', '1.0e-6', '1.0e-7'], 'indep_corner.txt', 'w')],
        ['indep_corner.txt'])
    add('near_one', 'quick', 'asymptotics', [(['near_one.py'], 'near_one.txt', 'w')], ['near_one.txt'])
    add('near_one_rates', 'quick', 'asymptotics', [(['near_one_rates.py'], None, 'w')], ['near_one_rates.txt'])
    add('uniqueness_jacobians', 'quick', 'asymptotics', [(['uniqueness_jacobians.py'], None, 'w')],
        ['uniqueness_jacobians.txt'])
    # ---- Section 6.3 (separation/)
    add('gap_check', 'quick', 'separation', [(['gap_check.py'], None, 'w')], ['gap_check.txt'])
    # ---- Section 7 (unrestricted/)
    add('obstruction_exact', 'quick', 'unrestricted', [(['obstruction_exact.py'], None, 'w')], ['obstruction_exact.txt'])
    add('adversary_lp', 'quick', 'unrestricted', [(['adversary_lp.py'], None, 'w')], ['adversary_lp.txt'])
    add('finite_n_exact', 'quick', 'unrestricted', [(['finite_n_exact.py'], None, 'w')], ['finite_n_exact.txt'])
    add('rhon_certify', 'quick', 'unrestricted', [(['rhon_certify.py'], None, 'w')], ['rhon_certify.txt'])
    add('rho3_certify', 'quick', 'unrestricted', [(['rho3_certify.py'], None, 'w')], ['rho3_certify.txt'])
    add('rhon_certify_78', 'full', 'unrestricted', [(['rhon_certify.py', '78', str(P)], None, 'w'),
                                                    (['rhon_certify.py', '78control'], None, 'w')],
        ['rhon_certify_78.txt'])
    add('conjecture_sweep', 'search', 'unrestricted', [(['conjecture_sweep.py'], None, 'w')], ['conjecture_sweep.txt'], 'info')
    add('conjecture_search', 'search', 'unrestricted', [(['conjecture_search.py'], None, 'w')], ['conjecture_search.txt'], 'info')
    add('band_finite_n', 'search', 'unrestricted', [(['band_finite_n.py'], None, 'w')], ['band_finite_n.txt'], 'info')
    add('finite_n_gap', 'search', 'unrestricted', [(['finite_n_gap.py'], None, 'w')], ['finite_n_gap.txt'], 'info')
    return S

TIERS = {'quick': {'quick'}, 'full': {'quick', 'full'}, 'search': {'quick', 'full', 'search'}}

TIMING = [(re.compile(r'\[\s*\d+(\.\d+)?\s*s\]'), '[T]'),
          (re.compile(r'\b\d+ LPs, \d+(\.\d+)?s\b'), 'N LPs, T'),
          (re.compile(r'elapsed \d+(\.\d+)?s'), 'elapsed T'),
          (re.compile(r'total time \d+(\.\d+)?s'), 'total time T'),
          (re.compile(r'\b\d+(\.\d+)?\s?(s|sec|seconds|min|minutes)\b'), 'T'),
          # point_values.py: the width of an enclosure depends on the floating-point starting point (the float curve)
          (re.compile(r'\(width \d+(\.\d+)?e[-+]\d+\)'), '(width W)')]

# lines that depend on whether a run reused a cached tube (certify/tube_*.json) or built it
CACHE_LINES = re.compile(r'^(round \d+: tested \d+, accepted \d+, split \d+|tube loaded from .*)$')

def normalize(text):
    out = []
    for line in text.splitlines():
        if CACHE_LINES.match(line.strip()):
            continue
        for rx, rep in TIMING:
            line = rx.sub(rep, line)
        out.append(line.rstrip())
    while out and not out[-1]:
        out.pop()
    return out

def compare(saved, new):
    """Returns (status, detail) for two output files."""
    if not os.path.exists(new):
        return 'MISSING', 'not written'
    if not os.path.exists(saved):
        return 'NEW', 'no saved output to compare with'
    if saved.endswith('.npy'):
        import numpy as np
        a, b = np.load(saved), np.load(new)
        if a.shape != b.shape:
            return 'DIFF', f'shapes {a.shape} vs {b.shape}'
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        return ('SAME' if d == 0 else 'DIFF'), f'max abs difference {d:.1e}'
    if saved.endswith('.json.gz'):
        a, b = json.load(gzip.open(saved)), json.load(gzip.open(new))
        pa = [(q.get('a'), q.get('b'), q.get('certified')) for q in a.get('pieces', [])]
        pb = [(q.get('a'), q.get('b'), q.get('certified')) for q in b.get('pieces', [])]
        return ('SAME' if pa == pb else 'DIFF'), f'pieces and certified flags {"agree" if pa == pb else "differ"}'
    A = normalize(open(saved, encoding='utf-8', errors='replace').read())
    B = normalize(open(new, encoding='utf-8', errors='replace').read())
    if A == B:
        return 'SAME', f'{len(A)} lines'
    diffs = [(i, x, y) for i, (x, y) in enumerate(zip(A, B)) if x != y]
    if len(A) != len(B):
        diffs.append((min(len(A), len(B)), f'<{len(A)} lines>', f'<{len(B)} lines>'))
    i, x, y = diffs[0]
    return 'DIFF', f'{len(diffs)} differing line(s); first at line {i + 1}:\n        saved: {x[:150]}\n        new:   {y[:150]}'

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tier', default='quick', choices=sorted(TIERS))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--workdir', default=os.path.join(HERE, '_reproduce'))
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    S = steps(args.procs)
    if args.list:
        for s in S:
            print(f"{s['tier']:7s} {s['name']:24s} {s['folder']:13s} -> {', '.join(s['outs']) or '(stdout only)'}")
        return
    todo = [s for s in S if (s['name'] in args.only if args.only else s['tier'] in TIERS[args.tier])]
    work = os.path.abspath(args.workdir)
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(HERE, work, ignore=shutil.ignore_patterns('.git', '.venv', '_reproduce*', '__pycache__', '*.pyc'))
    report = [f"reproduce.py --tier {args.tier} {'--only ' + ' '.join(args.only) if args.only else ''}".strip(),
              f"python {sys.version.split()[0]}; work directory {work}", '']
    def say(msg):
        print(msg, flush=True); report.append(msg)
    results = []
    for s in todo:
        cwd = os.path.join(work, s['folder'])
        env = dict(os.environ, **s['env'], PYTHONDONTWRITEBYTECODE='1')
        t0 = time.time(); ok = True; note = ''
        say(f"--- {s['name']} ({s['tier']}, {s['folder']})")
        for argv, target, mode in s['cmds']:
            out = open(os.path.join(cwd, target), mode) if target else subprocess.DEVNULL
            try:
                p = subprocess.run([PY] + argv, cwd=cwd, env=env, stdout=out if target else subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True)
            finally:
                if target:
                    out.close()
            if p.returncode != 0:
                ok = False; note = (p.stderr or '').strip().splitlines()[-1:] or ['']
                note = f"exit {p.returncode}: {note[0][:200]}"
                break
            if s['name'] == 'ivec_selftest' and 'OK' not in (p.stdout or ''):
                ok = False; note = 'self-test did not report OK'
        dt = time.time() - t0
        status = 'PASS' if ok else 'FAIL'
        details = []
        if ok:
            for o in s['outs']:
                st, d = compare(os.path.join(HERE, s['folder'], o), os.path.join(cwd, o))
                details.append(f"    {o}: {st} ({d})")
                if st in ('DIFF', 'MISSING') and s['compare'] == 'exact' and not o.endswith(('.npy', '.json.gz')):
                    status = 'DIFF'
        say(f"    {status} in {dt:.0f} s" + (f"; {note}" if note else ''))
        for d in details:
            say(d)
        results.append((s['name'], status, dt))
    say('\n=== summary ===')
    for name, st, dt in results:
        say(f"{st:5s} {name:24s} {dt:8.0f} s")
    n_ok = sum(st == 'PASS' for _, st, _ in results)
    say(f"{n_ok} of {len(results)} steps passed; total {sum(dt for *_, dt in results) / 60:.1f} min")
    open(os.path.join(work, 'reproduce_report.txt'), 'w').write('\n'.join(report) + '\n')
    sys.exit(0 if n_ok == len(results) else 1)

if __name__ == '__main__':
    main()
