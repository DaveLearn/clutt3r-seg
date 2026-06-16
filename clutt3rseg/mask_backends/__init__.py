"""2D instance-mask generation backends for Clutt3R-Seg.

The public release expects per-frame Grounded-SAM masks
(``data/instance_masks/mask_<frame>_<inst>.png``) to already exist; it does not
ship the detector. :mod:`clutt3rseg.mask_backends.grounded_sam` provides the
faithful Grounded-SAM step (GroundingDINO + SAM, prompt "object") so masks can be
generated for new sequences.
"""
