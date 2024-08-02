# Wukong-IV URDF Model Description

### [WuKongIV_boxColi.urdf](WuKongIV_boxColi.urdf)
Collision and visualization meshes are both composed of simple boxes and spheres.
This file is for structural error investigation. It also serves as a low-cost solution for visualization.

### [WuKongIV_fixedBase.urdf](WuKongIV_fixedBase.urdf)
This is a fixed-base version. It demonstrates the hanging performance on rack. 
Can be used when checking single joint response. 

### [WuKongIV_preciseMass.urdf](WuKongIV_preciseMass.urdf)
A floating-bsed model with the most precise mass and inertia measurement the lab can achieve by 2023.
The most commonly used robot model for simulation.
Acknowledgement: Wei Wandi

### [WuKongIV_sensor.urdf](WuKongIV_sensor.urdf)
Add IMU joint to the [floating-based file](WuKongIV_preciseMass.urdf). 
This file is designed to adapt to RaSim's new IMU sensor function.
These new functions are noe integrated into the production branch by Jul 2023.

### [WuKongIV_dexHand.urdf](WuKongIV_dexHand.urdf)
Modified based on [floating-based file](WuKongIV_preciseMass.urdf). 
The hand links are extra weighted to match the mass of human-like dexterous hands.
The meshes and DoFs around the hands haven't been changed.
These properties will be fully changed in the next robot version.


