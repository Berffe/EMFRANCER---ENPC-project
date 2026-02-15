# Starting Our Project

This is the repository our team (Felipe and Silas) will use to upload the most interesting contributions to the code and to help organize our progression. 

## JavaGenerator
This implementation may be very useful for you if you are using STAR-CCM+ to model your aircraft (kinda niche I know). But here we present 2 ways to automatically generate the sketches of your airfoils to then Loft into the actual body shape. After picking your design parameters (airfoil, top and front view of the aircraft and thickness of the trailling edge) in an Excel sheet or through a 3D modelling software (such as Catia), you put the information into a csv file (there is the Leading vs Trailling edge version and another Chord version) and can:

### Fine tune more aircraft parameters (ThickRatioInc.py)
Through this script, you are able to control the *thickness* throughout the wing-span, the *ratio* between your main airfoil and a symetric NACA one (interesting for Winglet modelling) and the angle of *incidence* in each section of your wing (control wing warp for example). The way you will control these parameters is the nodes of a PChipCurve, that is, a C1 continuous curve with the points you choose. Then you will:

### Generate the Java Macros
After that, you will click a buttom and have all your Java Macros to run in Star-CCM+! This way, the aerodynamics testing for different wing designs is way faster.

## Bibliographie
In this folder, we present some of the *a priori* research to help organize our implementations.
