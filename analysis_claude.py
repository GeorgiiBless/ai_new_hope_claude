"""Open clusters with Gaia DR3: Pleiades, Praesepe and M67.

1. Cone searches in gaiadr3.gaia_source_lite (astroquery.gaia), cached to data/.
2. Membership: HDBSCAN in standardized (pmra, pmdec, parallax).
3. Proper-motion and HR diagrams, cluster distances, turn-off based age ordering.
4. Optional: PARSEC isochrone age fit (distance and extinction held fixed).

Usage:  python analysis.py [--refresh] [--no-isochrones]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

G_LIMIT = 18.0
PARALLAX_ZERO_POINT = -0.017  # mas, global Gaia DR3 offset (Lindegren et al. 2021)
PARALLAX_SYSTEMATIC = 0.015  # mas, floor added to the cluster-parallax error
MIN_PROB = 0.1  # HDBSCAN membership probability cut inside the selected group
BRIGHT_G = 7.0  # RUWE is inflated for bright (saturated) stars: no RUWE cut brighter than this
FIT_MAG_LIMIT = 4.0  # isochrone fit uses M_G < 4: turn-off, subgiants, giants (lower MS is age-blind)
AV_SYSTEMATIC = 0.1  # refit at A_V +- this to get an extinction-systematic age range
A_G_PER_AV = 0.83  # approximate Gaia extinction coefficients for a solar-type star
E_BPRP_PER_AV = 0.42
RNG_SEED = 42

# Cone centres, radii and literature priors (A_V, [M/H]). The "lit" astrometry is
# only printed next to our results as a sanity check; it is never used for selection.
CLUSTERS = {
    "Pleiades": dict(ra=56.75, dec=24.12, radius=2.0, av=0.12, mh=0.0,
                     min_cluster_size=30, min_samples=10,
                     lit=dict(parallax=7.36, pmra=20.0, pmdec=-45.5)),
    "Praesepe": dict(ra=130.10, dec=19.67, radius=2.0, av=0.08, mh=0.15,
                     min_cluster_size=30, min_samples=10,
                     lit=dict(parallax=5.37, pmra=-36.0, pmdec=-12.9)),
    "M67": dict(ra=132.85, dec=11.81, radius=0.6, av=0.12, mh=0.0,
                min_cluster_size=30, min_samples=10,
                lit=dict(parallax=1.15, pmra=-11.0, pmdec=-2.9)),
}
COLOURS = {"Pleiades": "tab:blue", "Praesepe": "tab:orange", "M67": "tab:red"}

COLUMNS = ["source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error",
           "pmdec", "pmdec_error", "phot_g_mean_mag", "phot_bp_mean_mag",
           "phot_rp_mean_mag", "ruwe"]


# ---------------------------------------------------------------- data

def query_cluster(name, cfg, refresh=False):
    """Cone search in gaia_source_lite, cached as data/<name>.csv."""
    cache = DATA / f"{name.lower()}.csv"
    if cache.exists() and not refresh:
        print(f"[{name}] using cached {cache.relative_to(ROOT)}")
        return pd.read_csv(cache)

    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    adql = f"""
        SELECT {', '.join(COLUMNS)}
        FROM gaiadr3.gaia_source_lite
        WHERE 1 = CONTAINS(POINT('ICRS', ra, dec),
                           CIRCLE('ICRS', {cfg['ra']}, {cfg['dec']}, {cfg['radius']}))
          AND phot_g_mean_mag < {G_LIMIT}
    """
    print(f"[{name}] querying Gaia DR3 ({cfg['radius']} deg cone, G < {G_LIMIT}) ...")
    df = Gaia.launch_job_async(adql).get_results().to_pandas()
    df.columns = [c.lower() for c in df.columns]
    DATA.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"[{name}] {len(df)} sources saved to {cache.relative_to(ROOT)}")
    return df


def clean(df):
    """Quality cuts, parallax zero-point correction and colour."""
    finite = np.isfinite(df[["parallax", "pmra", "pmdec", "phot_g_mean_mag",
                             "phot_bp_mean_mag", "phot_rp_mean_mag"]]).all(axis=1)
    ruwe_ok = (df.ruwe < 1.4) | (df.phot_g_mean_mag < BRIGHT_G)
    good = finite & ruwe_ok & (df.parallax / df.parallax_error > 5)
    out = df[good].copy()
    out["plx"] = out.parallax - PARALLAX_ZERO_POINT
    out["bp_rp"] = out.phot_bp_mean_mag - out.phot_rp_mean_mag
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- membership

def select_members(name, df, cfg):
    """HDBSCAN in (pmra, pmdec, parallax); the cluster is the most compact group in PM."""
    X = StandardScaler().fit_transform(df[["pmra", "pmdec", "plx"]])
    hdb = HDBSCAN(min_cluster_size=cfg["min_cluster_size"],
                  min_samples=cfg["min_samples"], copy=True).fit(X)
    df["label"] = hdb.labels_
    df["prob"] = hdb.probabilities_

    groups = []
    for lab in set(hdb.labels_) - {-1}:
        g = df[df.label == lab]
        if len(g) >= 50:
            groups.append((np.hypot(g.pmra.std(), g.pmdec.std()), lab, len(g)))
    if not groups:
        raise RuntimeError(f"[{name}] HDBSCAN found no group with >= 50 stars")
    groups.sort()
    print(f"[{name}] HDBSCAN groups (pm dispersion, label, size): "
          + ", ".join(f"({s:.2f}, {l}, {n})" for s, l, n in groups))
    best = groups[0][1]

    df["member"] = (df.label == best) & (df.prob >= MIN_PROB)
    mem = df[df.member]
    lit = cfg["lit"]
    print(f"[{name}] {len(mem)} members | "
          f"pmra {mem.pmra.median():+.2f} (lit {lit['pmra']:+.1f}), "
          f"pmdec {mem.pmdec.median():+.2f} (lit {lit['pmdec']:+.1f}) mas/yr, "
          f"parallax {mem.plx.median():.3f} (lit {lit['parallax']:.2f}) mas")
    return df


def cluster_distance(mem):
    """Error-weighted mean parallax -> distance and distance modulus."""
    w = 1.0 / mem.parallax_error**2
    plx = np.sum(w * mem.plx) / np.sum(w)
    stat = max(np.sqrt(1.0 / np.sum(w)), mem.plx.std() / np.sqrt(len(mem)))
    plx_err = np.hypot(stat, PARALLAX_SYSTEMATIC)
    d = 1000.0 / plx
    return dict(parallax=plx, parallax_err=plx_err, distance=d,
                distance_err=d * plx_err / plx, dm=5 * np.log10(d) - 5)


def absolute_cmd(df, dm, av):
    """Dereddened colour and absolute G, using the cluster distance modulus for every star."""
    return df.bp_rp - E_BPRP_PER_AV * av, df.phot_g_mean_mag - dm - A_G_PER_AV * av


# ---------------------------------------------------------------- turn-off

def _turnoff_once(colour, mag, window, tol):
    """Ridge = running median over `window` stars sorted by M_G; turn-off = brightest
    ridge point within `tol` mag of the bluest ridge colour."""
    o = np.argsort(mag)
    c = np.median(sliding_window_view(colour[o], window), axis=1)
    m = np.median(sliding_window_view(mag[o], window), axis=1)
    i = np.flatnonzero(c <= c.min() + tol)[0]
    return c[i], m[i], i == 0


def find_turnoff(colour, mag, n_boot=500, window=15, tol=0.03):
    """Turn-off = where the members' main-sequence ridge reaches its bluest colour.

    Brighter than that the ridge turns red (subgiants/giants) or runs out of stars.
    Because the upper MS is nearly vertical, the brightest point within `tol` of the
    bluest colour is taken. Running medians keep blue stragglers and white dwarfs
    from pulling the ridge. Errors are bootstrap 16-84% half-ranges over members.
    """
    colour, mag = np.asarray(colour), np.asarray(mag)
    c0, m0, top = _turnoff_once(colour, mag, window, tol)
    rng = np.random.default_rng(RNG_SEED)
    boot = np.array([_turnoff_once(colour[k], mag[k], window, tol)[:2]
                     for k in rng.integers(0, len(mag), (n_boot, len(mag)))])
    half = lambda x: 0.5 * np.subtract(*np.percentile(x, [84, 16]))
    # top=True: the ridge is still bluest at the brightest stars -> only a bright limit
    return dict(to_colour=c0, to_colour_err=half(boot[:, 0]), to_mag=m0,
                to_mag_err=half(boot[:, 1]), to_is_top=bool(top))


# ---------------------------------------------------------------- isochrones

def load_parsec(mh):
    """PARSEC (CMD) isochrones in Gaia EDR3 passbands, log age 7.0-10.1, cached."""
    cache = DATA / f"parsec_MH{mh:+.2f}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    import ezpadova

    print(f"downloading PARSEC isochrones for [M/H] = {mh:+.2f} ...")
    iso = ezpadova.get_isochrones(logage=(7.0, 10.1, 0.05), MH=(mh, mh, 0.0),
                                  photsys_file="YBC_tab_mag_odfnew/tab_mag_gaiaEDR3.dat")
    iso = iso[["logAge", "MH", "Mini", "label", "Gmag", "G_BPmag", "G_RPmag"]]
    DATA.mkdir(exist_ok=True)
    iso.to_csv(cache, index=False)
    return iso


def densify(iso, step=0.01):
    """Resample one isochrone along its arc length, phase by phase (no jumps between phases)."""
    iso = iso[iso.label <= 7]  # PMS ... early AGB; drop TP-AGB and post-AGB
    pts = []
    for _, seg in iso.groupby((iso.label != iso.label.shift()).cumsum()):
        xy = np.column_stack([seg.G_BPmag - seg.G_RPmag, seg.Gmag])
        if len(xy) < 2:
            pts.append(xy)
            continue
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        t = np.linspace(0, s[-1], max(int(s[-1] / step), 2))
        pts.append(np.column_stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]))
    return np.vstack(pts)


def fit_age(mem, dm, av, iso_grid, clip=0.3, n_boot=200):
    """Grid fit in age with distance and A_V fixed.

    Cost = sum over members (M_G < FIT_MAG_LIMIT) of the clipped squared nearest-point
    distance to the isochrone in (colour, M_G); clipping absorbs binaries and stragglers.
    Statistical interval: bootstrap over members. Systematic range: refits at A_V +- AV_SYSTEMATIC.
    """
    ages = np.sort(iso_grid.logAge.unique())
    trees = [cKDTree(densify(iso_grid[np.isclose(iso_grid.logAge, a)])) for a in ages]

    def d2_matrix(a_v):
        colour, mag = map(np.asarray, absolute_cmd(mem, dm, a_v))
        sel = mag < FIT_MAG_LIMIT
        pts = np.column_stack([colour[sel], mag[sel]])
        return np.array([np.minimum(t.query(pts)[0] ** 2, clip**2) for t in trees])

    d2 = d2_matrix(av)
    best = ages[np.argmin(d2.sum(axis=1))]
    rng = np.random.default_rng(RNG_SEED)
    n = d2.shape[1]
    boot = [ages[np.argmin(d2[:, rng.integers(0, n, n)].sum(axis=1))] for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [16, 84])
    sys = [ages[np.argmin(d2_matrix(max(av + s, 0.0)).sum(axis=1))]
           for s in (-AV_SYSTEMATIC, AV_SYSTEMATIC)] + [best]
    return dict(log_age=best, age_myr=10**best / 1e6, age_lo_myr=10**lo / 1e6,
                age_hi_myr=10**hi / 1e6, age_sys_lo_myr=10 ** min(sys) / 1e6,
                age_sys_hi_myr=10 ** max(sys) / 1e6, n_fit=n)


# ---------------------------------------------------------------- plots

def legend(ax, **kw):
    """Legend with readable (enlarged, opaque) scatter markers."""
    leg = ax.legend(**kw)
    for h in leg.legend_handles:
        if isinstance(h, matplotlib.collections.PathCollection):
            h.set_sizes([25])
            h.set_alpha(1)
    return leg


def plot_pm(name, df):
    mem, fld = df[df.member], df[~df.member]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax1.scatter(fld.pmra, fld.pmdec, s=1, c="0.6", alpha=0.3, lw=0, label="field")
    ax1.scatter(mem.pmra, mem.pmdec, s=4, c=COLOURS[name], lw=0,
                label=f"members (N = {len(mem)})")
    # frame: field core plus the cluster
    lo = np.minimum(fld[["pmra", "pmdec"]].quantile(0.02), mem[["pmra", "pmdec"]].min()) - 3
    hi = np.maximum(fld[["pmra", "pmdec"]].quantile(0.98), mem[["pmra", "pmdec"]].max()) + 3
    ax1.set(xlim=(lo.pmra, hi.pmra), ylim=(lo.pmdec, hi.pmdec),
            xlabel=r"$\mu_{\alpha*}$ [mas/yr]", ylabel=r"$\mu_\delta$ [mas/yr]",
            title=f"{name}: proper motions")
    legend(ax1, loc="best")

    bins = np.linspace(0, max(df.plx.quantile(0.995), mem.plx.max()) + 0.5, 120)
    ax2.hist(fld.plx, bins, color="0.6", label="field")
    ax2.hist(mem.plx, bins, color=COLOURS[name], alpha=0.8, label="members")
    ax2.set(yscale="log", xlabel="parallax [mas]", ylabel="N", title=f"{name}: parallax")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"pm_{name.lower()}.png", dpi=150)
    plt.close(fig)


def plot_hr(name, df, res, iso=None):
    mem, fld = df[df.member], df[~df.member]
    fc, fm = absolute_cmd(fld, res["dm"], res["av"])
    mc, mm = absolute_cmd(mem, res["dm"], res["av"])
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    ax.scatter(fc, fm, s=1, c="0.75", alpha=0.3, lw=0, label="field (same distance)")
    ax.scatter(mc, mm, s=5, c=COLOURS[name], lw=0, label=f"members (N = {len(mem)})")
    if iso is not None:
        pts = densify(iso)
        ax.plot(pts[:, 0], pts[:, 1], ",", c="k")
        ax.plot([], [], "-", c="k", label=f"PARSEC {res['age_myr']:.0f} Myr, "
                                          f"[M/H] = {res['mh']:+.2f}")
    ax.errorbar(res["to_colour"], res["to_mag"], xerr=res["to_colour_err"],
                yerr=res["to_mag_err"], fmt="*", ms=14, mfc="gold", mec="k", ecolor="k",
                capsize=3, label="turn-off" + (" (bright limit)" if res["to_is_top"] else ""),
                zorder=5)
    ax.set(xlim=(-0.5, 3.5), ylim=(mm.max() + 0.5, min(mm.min(), -1) - 0.5),
           xlabel=r"$(G_{BP}-G_{RP})_0$", ylabel=r"$M_G$",
           title=f"{name}: d = {res['distance']:.0f} pc, A_V = {res['av']:.2f}")
    legend(ax, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"hr_{name.lower()}.png", dpi=150)
    plt.close(fig)


def plot_comparison(catalogs, results):
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    for name, df in catalogs.items():
        res = results[name]
        c, m = absolute_cmd(df[df.member], res["dm"], res["av"])
        ax.scatter(c, m, s=4, c=COLOURS[name], lw=0, alpha=0.7, label=name)
        ax.plot(res["to_colour"], res["to_mag"], "*", ms=18, mfc=COLOURS[name], mec="k")
    ax.invert_yaxis()
    ax.set(xlim=(-0.5, 3.5), xlabel=r"$(G_{BP}-G_{RP})_0$", ylabel=r"$M_G$",
           title="Turn-offs (stars): bluer and brighter = younger")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "hr_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-query Gaia instead of the cache")
    ap.add_argument("--no-isochrones", action="store_true", help="skip the PARSEC age fit")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    catalogs, results = {}, {}
    for name, cfg in CLUSTERS.items():
        df = select_members(name, clean(query_cluster(name, cfg, args.refresh)), cfg)
        mem = df[df.member]
        res = dict(cluster=name, n_members=int(len(mem)), av=cfg["av"], mh=cfg["mh"],
                   pmra=mem.pmra.median(), pmdec=mem.pmdec.median())
        res.update(cluster_distance(mem))
        res.update(find_turnoff(*absolute_cmd(mem, res["dm"], cfg["av"])))

        lit_d = 1000 / cfg["lit"]["parallax"]
        if abs(res["distance"] / lit_d - 1) > 0.1:
            print(f"WARNING [{name}]: distance {res['distance']:.0f} pc vs literature {lit_d:.0f} pc")
        if np.hypot(res["pmra"] - cfg["lit"]["pmra"], res["pmdec"] - cfg["lit"]["pmdec"]) > 1:
            print(f"WARNING [{name}]: member proper motion is > 1 mas/yr from literature")

        catalogs[name], results[name] = df, res
        mem.to_csv(OUT / f"members_{name.lower()}.csv", index=False)
        plot_pm(name, df)

    isos = {}
    if not args.no_isochrones:
        for name, df in catalogs.items():
            res = results[name]
            grid = load_parsec(res["mh"])
            res.update(fit_age(df[df.member], res["dm"], res["av"], grid))
            isos[name] = grid[np.isclose(grid.logAge, res["log_age"])]

    for name, df in catalogs.items():
        plot_hr(name, df, results[name], isos.get(name))
    plot_comparison(catalogs, results)

    # Bluer turn-off = more massive stars still on the MS = younger. The colour is the
    # primary ranking because the upper MS is nearly vertical (magnitude poorly defined).
    summary = pd.DataFrame(results.values())
    summary["age_rank"] = summary.to_colour.rank().astype(int)
    summary = summary.sort_values("age_rank")
    summary.to_csv(OUT / "summary.csv", index=False)

    cols = ["cluster", "n_members", "parallax", "parallax_err", "distance", "distance_err",
            "to_colour", "to_colour_err", "to_mag", "to_mag_err", "age_rank"]
    if not args.no_isochrones:
        cols += ["age_myr", "age_lo_myr", "age_hi_myr", "age_sys_lo_myr", "age_sys_hi_myr"]
    print("\n" + summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    top = summary.loc[summary.to_is_top, "cluster"].tolist()
    if top:
        print(f"note: the ridge of {', '.join(top)} is bluest at its brightest stars, "
              "so its turn-off magnitude is only a bright limit")
    measured = summary[~summary.to_is_top]
    if not measured.to_mag.is_monotonic_increasing:
        print("WARNING: measured turn-off magnitudes do not follow the same order as the colours")
    print("\nAge order (youngest -> oldest): " + " < ".join(summary.cluster))


if __name__ == "__main__":
    main()
