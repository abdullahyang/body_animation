The EHF is an evaluation dataset (i.e. for test-time) for methods performing holistic 3D human reconstruction.
The EHF dataset contains 100 frames manually curated by an expert annotator for confident 3D pseudo ground truth.
Each frame contains:
- filename: "*_img.jpg"    - RGB full-body image in JPG format.
- filename: "*_img.png"    - RGB full-body image in PNG format.
- filename: "*_scan.obj"   - 3D scan for this frame (multi-view stereo reconstruction).
- filename: "*_align.ply"  - Alignment of SMPL-X to the 3D scan, in the form of a 3D mesh (used as pseudo ground truth).
- filename: "*_2Djnt.json" - 2D joints estimated with OpenPose (from monocular RGB).
- filename: "*_2Djnt.png"  - Visualization of OpenPose 2D joints.
Both scans (.obj) and alignments (.ply) are in meters.

Additionally, the file 'EHF_camera.txt' contains the intrinsic and extrinsic parameters of the camera that captured the RGB frames of the EHF dataset.