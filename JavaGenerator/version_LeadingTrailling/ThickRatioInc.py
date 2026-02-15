from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from _makeProfile import schedule_pchip

## Import stuff
# Prends les courbes du borde d'attaque et du borde de fuite fait en CATIA
attaque = pd.read_csv("attaque_datum_aile2.csv", sep=';') 
class QC_att:
	at_X = attaque["X"].to_numpy()
	at_Y = -attaque["Y"].to_numpy()
	at_Z = attaque["Z"].to_numpy()
fuite = pd.read_csv("fuite_datum_aile2.csv", sep=';')
class QC_fuite:
	ft_X = fuite["X"].to_numpy()
	ft_Y = -fuite["Y"].to_numpy()
	ft_Z = fuite["Z"].to_numpy()

# Trouve le centre de rotation en Matlab
f = 0.25  # fraction from curve1 to curve2
class QC_cg:
	cg_X = (1 - f)*QC_att.at_X + f*QC_fuite.ft_X
	cg_Y = (1 - f)*QC_att.at_Y + f*QC_fuite.ft_Y
	cg_Z = (1 - f)*QC_att.at_Z + f*QC_fuite.ft_Z

###############################################################################################################
######### CONTROL CENTER ######################################################################################
Z_max = np.max(QC_att.at_Z*1e3)
zInc_cluster = [0, 130, 300, 450, 510, 540, Z_max]
inc_cluster = [6.22, 7.1, 6.6, 5.59, 1.5, 0, 0]	   

## ATTENTION: changer l'épaisseur (t_cluster) toujours avant que la proportion (w_cluster)!
## Si t_cluster != t_clark (11.72%) avec W(Z)=0, ça fait rien
zThick_cluster = [0, 515, 540, 550, Z_max]
t_cluster = [11.71/100, 11.71/100, 10.2/100, 8.5/100, 5/100]

zW_cluster = [0, 450, 500, 510, 540, Z_max]
w_cluster = np.array([0, 0, 0.5, 0.7, 1, 1])
###############################################################################################################
###############################################################################################################

class QC_funcs:
	INC_OF_Z = schedule_pchip(zInc_cluster, inc_cluster)
	T_OF_Z = schedule_pchip(zThick_cluster, t_cluster)
	W_OF_Z = schedule_pchip(zW_cluster, w_cluster)

def main():
	# # Graphique pour une visualization marrante
	# fig = plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot3D(QC_att.at_X, QC_att.at_Y, QC_att.at_Z, color='b')
	# ax.plot3D(QC_fuite.ft_X, QC_fuite.ft_Y, QC_fuite.ft_Z, color='r')
	# ax.plot3D(QC_cg.cg_X, QC_cg.cg_Y, QC_cg.cg_Z, color='k', linestyle='--')
	# ax.set_title('Courbe Guide Tracé')
	# ax.set_xlabel('X (mm)')
	# ax.set_ylabel('Y (mm)')
	# ax.set_zlabel('Z (mm)')
	# plt.show()

	# Sauvegarde la nouvelle courbe pour l'utiliser en Star-CCM+
	courbe_guide = pd.DataFrame({
		"X": QC_cg.cg_X,
		"Y": QC_cg.cg_Y,
		"Z": QC_cg.cg_Z,
	})
	# courbe_guide.to_csv("curve_mid_aile2.csv", index=False, header=False)
	
	# Block to take a look at the overall thickness, proportion between profiles, incidences and YZ configuration
	z_to_view = np.linspace(0, Z_max, 500)
	fig, ax1 = plt.subplots()
	ax1.plot(z_to_view, QC_funcs.T_OF_Z(z_to_view), label='Thickness', color='r')
	ax1.scatter(zThick_cluster, t_cluster)
	ax1.set_ylabel("Relative thickness (mm/mm)")
	ax2 = ax1.twinx()
	ax2.plot(QC_cg.cg_Z*1e3, QC_cg.cg_Y*1e3, label='Vue Face')
	ax2.set_ylabel("Y (mm)")	
	ax1.legend()
	ax2.legend()
	plt.title("Thickness Behaviour")
     
	fig, ax1 = plt.subplots()
	ax1.plot(z_to_view, 100*QC_funcs.W_OF_Z(z_to_view), label='Ratio', color='r')
	ax1.scatter(zW_cluster, w_cluster*100)
	ax1.set_ylabel("Proportion (%)")
	ax2 = ax1.twinx()
	ax2.plot(QC_cg.cg_Z*1e3, QC_cg.cg_Y*1e3, label='Vue Face')
	ax2.set_ylabel("Y (mm)")
	ax1.legend()
	ax2.legend()
	plt.title("Ratio Behaviour")
     
	fig, ax1 = plt.subplots()
	ax1.plot(z_to_view, QC_funcs.INC_OF_Z(z_to_view), label='Incidences', color='r')
	ax1.set_ylabel("Incidence (°)")
	ax1.scatter(zInc_cluster, inc_cluster)
	ax2 = ax1.twinx()
	ax2.plot(QC_cg.cg_Z*1e3, QC_cg.cg_Y*1e3, label='Vue Face')
	ax2.set_ylabel("Y (mm)")
	ax1.legend()
	ax2.legend()
	plt.title("Incidence Behaviour")
	plt.show()
	return

if __name__ == "__main__":
	main()