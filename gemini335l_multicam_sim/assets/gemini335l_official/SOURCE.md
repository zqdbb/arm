# Orbbec Gemini 335L official camera model

- Manufacturer: Orbbec
- Model: Gemini 335L and Gemini 336L shared enclosure
- Official repository: https://github.com/orbbec/OrbbecSDK_ROS2
- Branch: v2-main
- Mesh path: `orbbec_description/meshes/gemini335L_336L/base_link.STL`
- URDF path: `orbbec_description/urdf/gemini_335_L_336_L.urdf.xacro`
- License: Apache License 2.0
- Downloaded mesh SHA-256: `9de399ed805ddb004cacdf7656766d728d0e4fb9e9c83c70a3c12950c7db8303`

The original STL and Xacro are retained unchanged. The simulation creates a simplified PLY only for web display, applies the Xacro visual origin, converts the ROS camera frame to the depth optical frame, and then places one instance at each calibrated camera pose.

Verification against the supplied specification:

- Official STL envelope: approximately 124.02 x 29.03 x 27.00 mm
- Specification enclosure: 124 x 29 x 27 mm
- Official Xacro right IR offset: 95 mm from the left IR/depth origin
- Official Xacro RGB offset: 23.75 mm from the depth origin
