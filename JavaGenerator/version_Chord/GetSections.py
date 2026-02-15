import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

from _createJavaAll import generate_star_macro_java_plane_and_section, compute_outer_inner_and_te_links
from ThickRatioInc import DS_curve, QC_funcs

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

def interp_v(V, Z):
    V = np.asarray(V, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()

    order = np.argsort(Z)
    Zs = Z[order]
    Vs = V[order]

    Zu, idx = np.unique(Zs, return_index=True)
    Vu = Vs[idx]

    zmin, zmax = float(Zu[0]), float(Zu[-1])
    return PchipInterpolator(Zu, Vu, extrapolate=False)

def get_chorde(zed_mm, at_X, at_Y, at_Z, ft_X, ft_Z):
    atx = interp_v_at_z(at_X, at_Z, zed_mm, clamp=True)
    ftx = interp_v_at_z(ft_X, ft_Z, zed_mm, clamp=True)
    y_plane = interp_v_at_z(at_Y, at_Z, zed_mm, clamp=True)  

    chord = abs(atx - ftx)
    return chord, atx, y_plane

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
	epaisseur_wall = 0.1 		# Il faut choisir selon l'analyse d'impression 3D
	which_section = "All"		# Selon la section d'aile
	folder_name = "All"		# Name of the folder you are creating
	folder_location = r"C:\Users\Pipef\OneDrive\Academiques\Projet\Developpement\Modeles3D_Star\Courbe_Guide\version_Excel"
	Z_max = np.max(DS_curve.Z)
	interest_Zs = np.linspace(0, Z_max, 10)		# Changer selon la section!
	## Recommendation: Fuselage -> 0:120; Wing -> 125:450; Pre-Winglet -> 450: 500; Winglet -> 500:End

	for zed in interest_Zs:
		chord_mm = interp_v_at_z(DS_curve.cordes, DS_curve.Z, zed, clamp=True)
		x_plane = interp_v_at_z(DS_curve.X, DS_curve.Z, zed, clamp=True)
		y_plane = interp_v_at_z(DS_curve.Y, DS_curve.Z, zed, clamp=True)
		diedre = -get_diedre(zed, DS_curve.Y, DS_curve.Z)
		epaisseur_BF = interp_v_at_z(DS_curve.epaisseurs_BF, DS_curve.Z, zed, clamp=True)
		incidence_ang = QC_funcs.INC_OF_Z(zed)

		outer, inner, link_up, link_low = compute_outer_inner_and_te_links(
			thick_in_z = QC_funcs.T_OF_Z(zed),
			w_in_z =  QC_funcs.W_OF_Z(zed),
			chord_mm=chord_mm,
			ep_BF_mm=epaisseur_BF,
			ep_WALL_mm=epaisseur_wall,
			incidence_deg=incidence_ang,
			translate_x_mm=0,
			translate_y_mm=0,
			profils_dat_dir="ProfilsDAT",
		)

		java_name = '%s_WingSection_CutTE%d'%(which_section, zed)
		java = generate_star_macro_java_plane_and_section(
			macro_class=java_name,
			translation_m=(x_plane*1e-3, y_plane*1e-3, zed*1e-3),
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