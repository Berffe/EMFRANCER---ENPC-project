import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator, CubicSpline
import matplotlib.pyplot as plt

from _createJavaAll import generate_star_macro_java_plane_and_section, compute_outer_inner_and_te_links
from ThickRatioInc import QC_att, QC_fuite, QC_cg, QC_funcs

def parametrize_many_by_arclength(arrX, arrY, arrZ, **vars_1d):
    """
    arrX, arrY, arrZ: 1D arrays (N,) ordered along the curve
    vars_1d: any number of extra 1D arrays of length N (e.g., chord=..., BF=...)
    Returns:
      s: arc-length (M,)
      splines: dict mapping name -> CubicSpline(s, values)
               and also 'x','y','z' for coordinates.
    """
    x = np.asarray(arrX, float)
    y = np.asarray(arrY, float)
    z = np.asarray(arrZ, float)
    N = x.size
    if not (y.size == N and z.size == N):
        raise ValueError("X, Y, Z must have same length")

    extras = {}
    for k, v in vars_1d.items():
        vv = np.asarray(v, float)
        if vv.size != N:
            raise ValueError(f"{k} must have length {N}, got {vv.size}")
        extras[k] = vv

    p = np.column_stack((x, y, z))
    dp = np.diff(p, axis=0)
    ds = np.linalg.norm(dp, axis=1)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    # remove consecutive duplicate points so s is strictly increasing
    keep = np.concatenate(([True], ds > 0))
    s = s[keep]
    p = p[keep]
    for k in extras:
        extras[k] = extras[k][keep]

    splines = {
        "x": CubicSpline(s, p[:, 0], bc_type="natural"),
        "y": CubicSpline(s, p[:, 1], bc_type="natural"),
        "z": CubicSpline(s, p[:, 2], bc_type="natural"),
    }
    for k, vv in extras.items():
        # for "scalar" variables like chord/BF
        splines[k] = CubicSpline(s, vv, bc_type="natural")

    return s, splines

def eval_at_s(s_query, splines, keys=None):
    """
    Evaluate selected variables at s_query.
    Returns dict name -> array
    """
    s_query = np.asarray(s_query, float)
    if keys is None:
        keys = splines.keys()
    return {k: splines[k](s_query) for k in keys}

def interp_v_at_z(V, Z, z, clamp=True):
    V = np.asarray(V, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()

    order = np.argsort(Z)
    Zs = Z[order]
    Vs = V[order]

    Zu, idx = np.unique(Zs, return_index=True)
    Vu = Vs[idx]

    zmin, zmax = float(Zu[0]), float(Zu[-1])

    if clamp:
        z = float(np.clip(z, zmin, zmax))
    else:
        if not (zmin <= z <= zmax):
            raise ValueError(f"z={z} outside [{zmin}, {zmax}]")

    f = PchipInterpolator(Zu, Vu, extrapolate=False)
    return float(f(z))

def get_diedre(zed_mm, cg_Y, cg_Z, dz_mm=5.0):
    """
    Dihedral angle at zed_mm from the mid/quarter-chord curve.
    Uses atan2(dY, dZ), so never crashes when dZ ~ 0.
    """
    z0 = float(zed_mm)
    z1 = float(zed_mm - dz_mm)
    z2 = float(zed_mm + dz_mm)

    y1 = interp_v_at_z(cg_Y, cg_Z, z1, clamp=True)
    y2 = interp_v_at_z(cg_Y, cg_Z, z2, clamp=True)

    # z1/z2 are clamped internally, so compute actual clamped values too:
    z1c = float(np.clip(z1, np.min(cg_Z), np.max(cg_Z)))
    z2c = float(np.clip(z2, np.min(cg_Z), np.max(cg_Z)))

    dy = y2 - y1
    dz = z2c - z1c

    return float(np.degrees(np.arctan2(dy, dz)))

def main():
	#################################################################################################################################
    ###### CONTROL CENTER ###########################################################################################################
	epaisseur_wall = 0.1 		# Il faut choisir selon l'analyse d'impression 3D
	epaisseur_BF = 1			# Selon l'analyse a priori
	which_section = "Pre_Winglet"		# Selon la section d'aile
	folder_name = "Pre_Winglet"		# Name of the folder you are creating
	folder_location = r"C:\Users\Pipef\OneDrive\Academiques\Projet\Developpement\Modeles3D_Star\Final\AileCourte"

	chords = abs(QC_att.at_X - QC_fuite.ft_X)
	s, spl = parametrize_many_by_arclength(
		QC_cg.cg_X, QC_cg.cg_Y, QC_cg.cg_Z, chord=chords
	)
	
	## Recommendation: Fuselage -> 0:180; Wing -> 185:500; Pre-Winglet -> 505:640; Winglet -> 645:End
	# first_s = s[0]; last_s = s[-1]
	S_params = np.linspace(645, s[-1], 10)
	S_curve = eval_at_s(S_params, spl, keys=["x","y","z","chord"])     # (500,3)
	Xns, Yns, Zns = S_curve["x"], S_curve["y"], S_curve["z"]
	chords_n = S_curve["chord"]

	for which in range(len(S_params)):
		chord_mm = chords_n[which]
		x_plane = Xns[which]
		y_plane = Yns[which]
		z_plane = Zns[which]
		diedre = -get_diedre(z_plane, Yns, Zns)
		incidence_ang = QC_funcs.INC_OF_Z(z_plane)
		
		incidence_ang = QC_funcs.INC_OF_Z(z_plane)
		outer, inner, link_up, link_low = compute_outer_inner_and_te_links(
			thick_in_z = QC_funcs.T_OF_Z(z_plane),
			w_in_z = QC_funcs.W_OF_Z(z_plane),
			chord_mm=chord_mm,
			ep_BF_mm=epaisseur_BF,
			ep_WALL_mm=epaisseur_wall,
			incidence_deg=incidence_ang,
			translate_x_mm=0,
			translate_y_mm=0,
			profils_dat_dir="ProfilsDAT",
		)

		java_name = '%s_WingSection_CutTE%d'%(which_section, S_params[which])
		java = generate_star_macro_java_plane_and_section(
			macro_class=java_name,
			translation_m=(x_plane*1e-3, y_plane*1e-3, z_plane*1e-3),
			dihedral_deg=diedre,
			outer_xy_m=outer,
			inner_xy_m=inner,
			te_link_up=link_up,
			te_link_low=link_low,
		)
          
		out_dir = Path(folder_location)
		out_dir = out_dir / folder_name
		out_dir.mkdir(parents=True, exist_ok=True)
		java_comp = '%s.java'%(java_name)
		out_path = out_dir / java_comp
		out_path.write_text(java, encoding="utf-8")
		print(f"Wrote: {out_path.resolve()}")

if __name__ == '__main__':
	main()