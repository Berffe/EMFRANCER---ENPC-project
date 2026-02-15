import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator

def load_airfoil_dat(dat_path: Path) -> np.ndarray:
    lines = dat_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    pts = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            pts.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if len(pts) < 2:
        raise ValueError(f"Not enough points in {dat_path}")
    return np.asarray(pts, dtype=float)  # Nx2 normalized

def naca00xx_half_thickness(x, t, closed_te=True):
    """
    Symmetric NACA 00xx half-thickness y_t(x) for normalized chord (c=1).
    x: array in [0,1]
    t: thickness ratio (e.g. 0.12 for 12%)
    """
    x = np.asarray(x, float)
    a4 = -0.1015 if closed_te else -0.1036
    yt = 5.0 * t * (
        0.2969*np.sqrt(np.clip(x, 0, 1))
        - 0.1260*x
        - 0.3516*x**2
        + 0.2843*x**3
        + a4*x**4
    )
    return yt

def airfoil_loop_to_surfaces(xy):
    """
    xy: (N,2) array of airfoil coordinates in a closed loop.
    Returns: x_u, y_u, x_l, y_l (each monotone in x from 0->1)
    """
    xy = np.asarray(xy, float)
    x, y = xy[:,0], xy[:,1]
    i_le = int(np.argmin(x))  # leading edge index (min x)

    upper = xy[:i_le+1]      # TE -> LE
    lower = xy[i_le:]        # LE -> TE

    # Make both run LE->TE with increasing x
    upper = upper[::-1]      # LE -> TE
    # lower already LE -> TE typically

    # Sort by x just in case and unique-ify x for interpolation
    def _mono(arr):
        xs = arr[:,0]; ys = arr[:,1]
        order = np.argsort(xs)
        xs = xs[order]; ys = ys[order]
        xu, idx = np.unique(xs, return_index=True)
        yu = ys[idx]
        return xu, yu

    x_u, y_u = _mono(upper)
    x_l, y_l = _mono(lower)
    return x_u, y_u, x_l, y_l

def surfaces_on_grid(x_grid, x_u, y_u, x_l, y_l):
    """
    Interpolate upper/lower surfaces onto a shared x_grid in [0,1].
    """
    xg = np.asarray(x_grid, float)
    yu = np.interp(xg, x_u, y_u)
    yl = np.interp(xg, x_l, y_l)
    return yu, yl

def camber_thickness(yu, yl):
    yc = 0.5*(yu + yl)
    yt = 0.5*(yu - yl)
    return yc, yt

def max_thickness_ratio(yu, yl):
    return float(np.max(yu - yl))  # since chord normalized to 1

def blend_clark_to_naca(
    clark_xy, 
    w,               # 0..1  (0=Clark, 1=NACA)
    t_naca,          # thickness ratio for NACA (e.g. 0.08)
    x_grid=None,
    closed_te=True
):
    """
    Returns blended airfoil loop coords (TE->upper->LE->lower->TE), normalized chord.
    """
    w = float(np.clip(w, 0.0, 1.0))
    if x_grid is None:
        # cosine spacing gives better LE resolution
        beta = np.linspace(0, np.pi, 200)
        x_grid = 0.5*(1 - np.cos(beta))
    xg = np.asarray(x_grid, float)

    # Clark surfaces
    x_u, y_u, x_l, y_l = airfoil_loop_to_surfaces(clark_xy)
    yu_c, yl_c = surfaces_on_grid(xg, x_u, y_u, x_l, y_l)
    yc_c, yt_c = camber_thickness(yu_c, yl_c)

    # NACA surfaces (symmetric => camber 0)
    yt_n = naca00xx_half_thickness(xg, t_naca, closed_te=closed_te)
    yc_n = np.zeros_like(xg)

    # Blend in camber/thickness space
    yc_b = (1 - w)*yc_c + w*yc_n
    yt_b = (1 - w)*yt_c + w*yt_n

    yu_b = yc_b + yt_b
    yl_b = yc_b - yt_b

    # Build loop: TE->upper->LE->lower->TE (common DAT order)
    upper = np.column_stack([xg[::-1], yu_b[::-1]])   # TE->LE
    lower = np.column_stack([xg[1:],  yl_b[1:]])      # LE->TE (skip duplicate LE)
    loop = np.vstack([upper, lower])
    return loop

def choose_naca_thickness_percent(
    clark_xy,
    w,
    t_des,               # desired max thickness ratio (0.12 etc.) at this station
    t_min=0.02,
    t_max=0.20
):
    """
    Returns xx (percent thickness) to use in NACA00xx so that
    blended thickness follows t_des as smoothly as you define it.
    """
    w = float(np.clip(w, 0.0, 1.0))

    # Compute Clark thickness ratio once (on a decent grid)
    beta = np.linspace(0, np.pi, 400)
    xg = 0.5*(1 - np.cos(beta))

    x_u, y_u, x_l, y_l = airfoil_loop_to_surfaces(clark_xy)
    yu_c, yl_c = surfaces_on_grid(xg, x_u, y_u, x_l, y_l)
    t_clark = max_thickness_ratio(yu_c, yl_c)

    t_des = float(t_des)

    if w < 1e-9:
        # you're basically Clark; NACA choice doesn't matter
        t_naca = t_clark
    else:
        t_naca = (t_des - (1.0 - w)*t_clark) / w

    t_naca = float(np.clip(t_naca, t_min, t_max))
    return 100.0 * t_naca

def schedule_pchip(z_knots, t_knots):
    """
    Build a monotone-friendly thickness schedule t(z) using PCHIP.

    Parameters
    ----------
    z_knots : array-like
        Span stations in mm (must be increasing).
    t_knots : array-like
        Thickness ratios at those stations (e.g., 0.12 for 12%).
        If you want monotone decrease, make these decreasing.

    Returns
    -------
    t_of_z : callable
        Function t(z) that returns thickness ratio for any z in range.
    """
    z = np.asarray(z_knots, dtype=float)
    t = np.asarray(t_knots, dtype=float)

    order = np.argsort(z)
    z = z[order]
    t = t[order]

    # Remove duplicate z's if any
    zu, idx = np.unique(z, return_index=True)
    tu = t[idx]

    f = PchipInterpolator(zu, tu, extrapolate=False)

    def t_of_z(zq):
        zq = np.asarray(zq, dtype=float)
        # clamp to avoid out-of-range NaNs
        zq = np.clip(zq, zu[0], zu[-1])
        return f(zq)

    return t_of_z

def schedule_smoother(z_knots, t_knots):
    """
    Build a monotone-friendly thickness schedule t(z) using PCHIP.

    Parameters
    ----------
    z_knots : array-like
        Span stations in mm (must be increasing).
    t_knots : array-like
        Thickness ratios at those stations (e.g., 0.12 for 12%).
        If you want monotone decrease, make these decreasing.

    Returns
    -------
    t_of_z : callable
        Function t(z) that returns thickness ratio for any z in range.
    """
    z = np.asarray(z_knots, dtype=float)
    t = np.asarray(t_knots, dtype=float)

    order = np.argsort(z)
    z = z[order]
    t = t[order]

    # Remove duplicate z's if any
    zu, idx = np.unique(z, return_index=True)
    tu = t[idx]

    f = Akima1DInterpolator(zu, tu)

    def t_of_z(zq):
        zq = np.asarray(zq, dtype=float)
        # clamp to avoid out-of-range NaNs
        zq = np.clip(zq, zu[0], zu[-1])
        return f(zq)

    return t_of_z

def generate_airfoil(t_des, w):
	profils_dat_dir = "ProfilsDAT"
	profils_dat_dir = Path(profils_dat_dir)
	clark_xy = load_airfoil_dat(profils_dat_dir / "CLARK_YS.dat")

	xx = choose_naca_thickness_percent(clark_xy, w, t_des)
	t_naca = xx/100.0

	return blend_clark_to_naca(clark_xy, w=w, t_naca=t_naca)	

if __name__ == "__main__":
	profils_dat_dir = "ProfilsDAT"
	profils_dat_dir = Path(profils_dat_dir)
	clark_xy = load_airfoil_dat(profils_dat_dir / "CLARK_YS.dat")

	z_cluster = [450, 500, 515, 530]
	t_cluster = [11.71/100, 9/100, 7/100, 5/100]
	w_cluster = [0, 0.3, 0.6, 1]
    
	t_of_z = schedule_pchip(z_cluster, t_cluster)
	w_of_z = schedule_pchip(z_cluster, w_cluster)
	zeds = np.linspace(0, 450, 10)
	w_s = []
	nacas = []
	fig, ax = plt.subplots()
	for zed in zeds:
		t_des = t_of_z(zed)
		w = w_of_z(zed)

		xx = choose_naca_thickness_percent(clark_xy, w, t_des)
		t_naca = xx/100.0

		w_s.append(w)
		nacas.append(t_des)
		blended_xy = blend_clark_to_naca(clark_xy, w=w, t_naca=t_naca)	
		# print(blended_xy)
		ax.plot(blended_xy[:, 0], blended_xy[:, 1])

	ax.set_xlim(0, 1)
	ax.set_ylim(-0.5, 0.5)

	plt.figure()
	plt.plot(zeds, nacas)
	plt.title("Thickness")
	plt.figure()
	plt.plot(zeds, w_s)
	plt.title("Ratio")
	plt.show()