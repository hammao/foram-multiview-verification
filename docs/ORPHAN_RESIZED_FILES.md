# Orphan files in `processed/resized/`

A directory listing of `processed/resized/` finds **10,496** PNGs. `views_resized.csv`
lists **8,219**. The 2,277 files that are on disk but not in the manifest are not part of
any split and must not be counted as a third dataset size.

They are leftovers from earlier discovery settings:

- `*_rotated90.png` — produced when `gif_include_rotated90` was true; current configs set
  that flag to false because a 90° GIF frame is the same view, not a new one
- stale Z-stack slices whose names no longer match the current manifest after a
  re-standardisation (`resize.overwrite: true` regenerates listed files and never deletes
  unlisted ones)

The modelling denominator is the manifest, not the directory listing.
