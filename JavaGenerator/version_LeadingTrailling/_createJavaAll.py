from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from _makeProfile import generate_airfoil

# -----------------------------
# Helpers copied/adapted from your _wingToJava.py logic
# -----------------------------
def _matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    if step == 0:
        raise ValueError("step cannot be 0")
    n = int(math.floor((stop - start) / step)) + 1
    if n <= 0:
        return np.array([start], dtype=float)
    arr = start + step * np.arange(n, dtype=float)
    if abs(arr[-1] - stop) <= 1e-12 * max(1.0, abs(stop)):
        arr[-1] = stop
    return arr

def _spline_interp(x: np.ndarray, y: np.ndarray, delta_x: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xx = _matlab_colon(x[0], delta_x, x[-1])
    cs = CubicSpline(x, y)  # MATLAB-like "spline"
    yy = cs(xx)
    return xx.reshape(-1, 1), yy.reshape(-1, 1)


def _split_extrados_intrados(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = A[:, 0]
    y = A[:, 1]
    x0 = int(np.argmin(x))
    X_extr = x[: x0 + 1][::-1].copy()
    Y_extr = y[: x0 + 1][::-1].copy()
    X_intr = x[x0:].copy()
    Y_intr = y[x0:].copy()
    return X_extr, Y_extr, X_intr, Y_intr


def _find_bf_index(y_extr: np.ndarray, y_intr: np.ndarray, ep_bf: float) -> int:
    y_extr = y_extr.ravel()
    y_intr = y_intr.ravel()
    i = len(y_extr) - 1
    l_bf = y_extr[i] - y_intr[i]
    while l_bf < ep_bf:
        i -= 1
        if i < 0:
            raise RuntimeError("Could not find a BF location satisfying thickness constraint.")
        l_bf = y_extr[i] - y_intr[i]
    return i


def _rotate_about_quarter_chord(xy: np.ndarray, angle_deg: float, deplace_x_m: float, deplace_y_m: float) -> np.ndarray:
    ang = math.radians(angle_deg)
    c = math.cos(ang)
    s = math.sin(ang)

    x = xy[:, 0].copy() - deplace_x_m
    y = xy[:, 1].copy() + deplace_y_m

    X = x * c + y * s
    Y = -x * s + y * c

    X = X + deplace_x_m
    Y = Y - deplace_y_m
    return np.column_stack([X, Y])


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


def compute_outer_inner_and_te_links(
	thick_in_z: float,
	w_in_z: float,
	chord_mm: float,
	ep_BF_mm: float,
	ep_WALL_mm: float,
	incidence_deg: float,
	translate_x_mm: float,
	translate_y_mm: float,
	profils_dat_dir: str | Path = "ProfilsDAT",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Returns:
		outer_xy_m (N x 2)
		inner_xy_m (M x 2)
		te_link_up (2 x 2)
		te_link_low (2 x 2)
	Coordinates are in meters in the sketch plane.
	"""
	profils_dat_dir = Path(profils_dat_dir)

	ep_BF_Int_mm = ep_BF_mm + 2.0 * ep_WALL_mm

	# Generates Airfoil
	Aile_norm = generate_airfoil(thick_in_z, w_in_z)

    # ---- Outer (Ext)
	X_extr_init, Y_extr_init, X_intr_init, Y_intr_init = _split_extrados_intrados(Aile_norm)

	x_extr = X_extr_init * chord_mm
	y_extr = Y_extr_init * chord_mm
	x_intr = X_intr_init * chord_mm
	y_intr = Y_intr_init * chord_mm

	delta_x = 0.001  # mm
	x_extr_i, y_extr_i = _spline_interp(x_extr, y_extr, delta_x)
	x_intr_i, y_intr_i = _spline_interp(x_intr, y_intr, delta_x)

	debut = _find_bf_index(y_extr_i, y_intr_i, ep_BF_mm)
	X_BF = float(x_intr_i[debut, 0])

	chord_new_mm = float(chord_mm)
	while X_BF < chord_mm:
		chord_new_mm = chord_new_mm + (chord_mm - X_BF)

		x_extr = X_extr_init * chord_new_mm
		y_extr = Y_extr_init * chord_new_mm
		x_intr = X_intr_init * chord_new_mm
		y_intr = Y_intr_init * chord_new_mm

		x_extr_i, y_extr_i = _spline_interp(x_extr, y_extr, delta_x)
		x_intr_i, y_intr_i = _spline_interp(x_intr, y_intr, delta_x)

		debut = _find_bf_index(y_extr_i, y_intr_i, ep_BF_mm)
		X_BF = float(x_intr_i[debut, 0])

	chord_m = chord_mm / 1000.0
	chord_new_m = chord_new_mm / 1000.0

	outer = Aile_norm * chord_new_m  # meters

	# Keep range around original chord (same logic as your script)
	xA = outer[:, 0]
	idx_lt = np.where(xA < chord_m)[0]
	idx_gt = np.where(xA > chord_m)[0]
	if idx_lt.size == 0 or idx_gt.size == 0:
		raise RuntimeError("Unexpected profile x-range.")
	i0 = int(idx_lt.min())
	i1 = int(idx_gt.max()) - 1
	outer = outer[i0 : i1 + 1, :]

	idx_lt2 = np.where(outer[:, 0] < chord_m)[0]
	nb_max = int(idx_lt2.max())
	outer = outer[: nb_max + 1, :]

	# Insert TE points (gives TE thickness = ep_BF)
	y_extr_te_m = float(y_extr_i[debut, 0]) / 1000.0
	y_intr_te_m = float(y_intr_i[debut, 0]) / 1000.0
	outer = np.vstack([
		np.array([[chord_m, y_extr_te_m]]),
		outer,
		np.array([[chord_m, y_intr_te_m]])
	])

	# Save for inner build (TOTO in your script)
	TOTO = outer.copy()

	# Rotate about quarter chord + translate (incidence)
	deplace_x_m = (chord_mm / 4.0) / 1000.0
	deplace_y_m = 0.0
	outer = _rotate_about_quarter_chord(outer, incidence_deg, deplace_x_m, deplace_y_m)

	tx_m = translate_x_mm / 1000.0
	ty_m = translate_y_mm / 1000.0
	outer[:, 0] += tx_m
	outer[:, 1] += ty_m

	# ---- Inner (Int)
	X_extr_init, Y_extr_init, X_intr_init, Y_intr_init = _split_extrados_intrados(TOTO)

	x_extr = X_extr_init
	y_extr = Y_extr_init
	x_intr = X_intr_init
	y_intr = Y_intr_init

	delta_x_int = 0.0001  # meters
	x_extr_i, y_extr_i = _spline_interp(x_extr, y_extr, delta_x_int)
	x_intr_i, y_intr_i = _spline_interp(x_intr, y_intr, delta_x_int)

	ep_BF_Int_m = ep_BF_Int_mm / 1000.0
	debut_int = _find_bf_index(y_extr_i, y_intr_i, ep_BF_Int_m)
	X_BF_int = float(x_intr_i[debut_int, 0])

	x_TOTO = TOTO[:, 0].copy()
	y_TOTO = TOTO[:, 1].copy()
	x_TITI = x_TOTO.copy()
	y_TITI = y_TOTO.copy()

	x_extr_flat = x_extr_i.ravel()
	x_intr_flat = x_intr_i.ravel()
	y_extr_flat = y_extr_i.ravel()
	y_intr_flat = y_intr_i.ravel()

	idx_extr = np.where(x_extr_flat < X_BF_int)[0]
	idx_intr = np.where(x_intr_flat < X_BF_int)[0]
	if idx_extr.size == 0 or idx_intr.size == 0:
		raise RuntimeError("Could not build interior trimming indices near BF.")

	x_extr_cut = float(x_extr_flat[idx_extr.max()])
	y_extr_cut = float(y_extr_flat[idx_extr.max()])
	x_intr_cut = float(x_intr_flat[idx_intr.max()])
	y_intr_cut = float(y_intr_flat[idx_intr.max()])

	while x_TITI[0] > x_extr_cut:
		x_TITI = x_TITI[1:]
		y_TITI = y_TITI[1:]
		if x_TITI.size == 0:
			raise RuntimeError("Interior trimming removed all points (head).")

	while x_TITI[-1] > x_intr_cut:
		x_TITI = x_TITI[:-1]
		y_TITI = y_TITI[:-1]
		if x_TITI.size == 0:
			raise RuntimeError("Interior trimming removed all points (tail).")

	x_TITI = np.concatenate([[x_extr_cut], x_TITI, [x_intr_cut]])
	y_TITI = np.concatenate([[y_extr_cut], y_TITI, [y_intr_cut]])

	x = x_TITI
	y = y_TITI

	d = ep_WALL_mm / 1000.0
	dx = np.gradient(x)
	dy = np.gradient(y)
	norme = np.sqrt(dx * dx + dy * dy)
	norme = np.where(norme == 0.0, 1e-16, norme)

	n_int_x = -dy / norme
	n_int_y = dx / norme

	x_int = x + d * n_int_x
	y_int = y + d * n_int_y

	inner = np.column_stack([x_int, y_int])
	inner = _rotate_about_quarter_chord(inner, incidence_deg, deplace_x_m, deplace_y_m)
	inner[:, 0] += tx_m
	inner[:, 1] += ty_m

	# ---- TE links (close section at tail)
	outer_te_up = outer[0, :]
	outer_te_low = outer[-1, :]
	inner_start = inner[0, :]
	inner_end = inner[-1, :]

	te_link_up = np.vstack([outer_te_up, outer_te_low])
	te_link_low = np.vstack([inner_end, inner_start])

	return outer, inner, te_link_up, te_link_low


# -----------------------------
# Java macro writing
# -----------------------------
def _flatten_xy(points_xy_m: np.ndarray) -> str:
    chunks = []
    for x, y in points_xy_m:
        chunks.append(f"{x:.12f}, {y:.12f}")
    return ",\n          ".join(chunks)


def generate_star_macro_java_plane_and_section(
    macro_class: str,
    translation_m: Tuple[float, float, float],
    dihedral_deg: float,
    outer_xy_m: np.ndarray,
    inner_xy_m: np.ndarray,
    te_link_up: np.ndarray,
    te_link_low: np.ndarray,
) -> str:
    tx, ty, tz = translation_m

    pts_outer = _flatten_xy(outer_xy_m)
    pts_inner = _flatten_xy(inner_xy_m)
    pts_link_up = _flatten_xy(te_link_up)     # only 2 points => straight segment
    pts_link_low = _flatten_xy(te_link_low)   # only 2 points => straight segment

    return f"""// STAR-CCM+ macro: {macro_class}.java
// Generated by Python
package macro;

import java.util.*;
import star.common.*;
import star.base.neo.*;
import star.cadmodeler.*;
import star.vis.*;

public class {macro_class} extends StarMacro {{

  public void execute() {{
    execute0();
  }}

  private void execute0() {{

    Simulation simulation_0 = getActiveSimulation();

    CadModel cadModel_0 =
      ((CadModel) simulation_0.get(SolidModelManager.class).getObject("3D-CAD Model 1"));

    cadModel_0.allowMakingPartDirty(false);

    Scene scene_0 = simulation_0.getSceneManager().getScene("3D-CAD View 1");
    SceneUpdate sceneUpdate_0 = scene_0.getSceneUpdate();
    HardcopyProperties hardcopyProperties_0 = sceneUpdate_0.getHardcopyProperties();

    CanonicalSketchPlane canonicalSketchPlane_0 =
      ((CanonicalSketchPlane) cadModel_0.getFeature("XY"));

    TransformSketchPlane transformSketchPlane_0 =
      cadModel_0.getFeatureManager().createPlaneByTransformation(canonicalSketchPlane_0);

    transformSketchPlane_0.setRefSketchPlane(canonicalSketchPlane_0);

    Units units_m = ((Units) simulation_0.getUnitsManager().getObject("m"));
    transformSketchPlane_0.getTranslationVector().setUnits(units_m);
    transformSketchPlane_0.getTranslationVector().setComponents({tx:.12f}, {ty:.12f}, {tz:.12f});

    Units units_deg = ((Units) simulation_0.getUnitsManager().getObject("deg"));
    transformSketchPlane_0.getAngle().setUnits(units_deg);
    // Dihedral as rotation about X
    transformSketchPlane_0.getAngle().setComponents({dihedral_deg:.12f}, 0.0, 0.0);

    transformSketchPlane_0.setIsBodyGroupCreation(false);
    cadModel_0.getFeatureManager().markDependentNotUptodate(transformSketchPlane_0);

    cadModel_0.allowMakingPartDirty(true);
    transformSketchPlane_0.markFeatureForEdit();
    cadModel_0.getFeatureManager().execute(transformSketchPlane_0);

    // --- Creates new sketch
    Sketch sketch_0 = cadModel_0.getFeatureManager().createSketch(transformSketchPlane_0);
    sketch_0.setAutoPreview(true);

    cadModel_0.allowMakingPartDirty(false);
    cadModel_0.getFeatureManager().startSketchEdit(sketch_0);

    // --- Outer contour with trailing-edge links (Ext) ---
    sketch_0.createSpline(false, null, false, null,
      new DoubleVector(new double[] {{
          {pts_outer}
      }}));
    
    // Link
    sketch_0.createSpline(false, null, false, null,
      new DoubleVector(new double[] {{
          {pts_link_up}
      }}));
      
    cadModel_0.allowMakingPartDirty(true);
    cadModel_0.getFeatureManager().stopSketchEdit(sketch_0, true);
    
    // --- Creates new sketch
	Sketch sketch_1 = cadModel_0.getFeatureManager().createSketch(transformSketchPlane_0);
    sketch_1.setAutoPreview(true);
    
    cadModel_0.allowMakingPartDirty(false);
    cadModel_0.getFeatureManager().startSketchEdit(sketch_1);
    
    // --- Inner contour with trailing-edge links (Int) ---
    sketch_1.createSpline(false, null, false, null,
      new DoubleVector(new double[] {{
          {pts_inner}
      }}));
      
    // Link
    sketch_1.createSpline(false, null, false, null,
      new DoubleVector(new double[] {{
          {pts_link_low}
      }}));
    cadModel_0.allowMakingPartDirty(true);
    cadModel_0.getFeatureManager().stopSketchEdit(sketch_1, true);
    cadModel_0.getFeatureManager().updateModelAfterFeatureEdited(sketch_1, null);
  }}
}}
"""

def main():
    # ---- Single station example (you'll automate later) ----
    # Plane center in meters:
    X_m, Y_m, Z_m = 0.0, 0.0, 0.200

    # Plane dihedral rotation about X (deg):
    dihedral_deg = 5.0

    # Profile params:
    t_in_z = 10/100
    w_in_z = 50/100
    chord_mm = 100.0
    incidence_deg = 10.0
    ep_BF_mm = 1.0
    ep_WALL_mm = 0.8

    # Optional in-plane translation of the profile (mm):
    translate_x_mm = 0.0
    translate_y_mm = 0.0

    outer, inner, link_up, link_low = compute_outer_inner_and_te_links(
        t_des=t_in_z,
        w = w_in_z,
        chord_mm=chord_mm,
        ep_BF_mm=ep_BF_mm,
        ep_WALL_mm=ep_WALL_mm,
        incidence_deg=incidence_deg,
        translate_x_mm=translate_x_mm,
        translate_y_mm=translate_y_mm,
        profils_dat_dir="ProfilsDAT",
    )

    java = generate_star_macro_java_plane_and_section(
        macro_class="WingSection_CutTE",
        translation_m=(X_m, Y_m, Z_m),
        dihedral_deg=dihedral_deg,
        outer_xy_m=outer,
        inner_xy_m=inner,
        te_link_up=link_up,
        te_link_low=link_low,
    )

    out_dir = Path("ProfilsJAVA")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "WingSection_CutTE.java"
    out_path.write_text(java, encoding="utf-8")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
