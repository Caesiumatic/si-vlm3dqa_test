# Data decisions before any model metrics

2026-09-25.

## Source

Explicit-NeRF-QA HF revision `01db6a2db25fe0d66c63bc4d63c5d6efaa401547`.
Published zip: 21,881,818,838 bytes. Central directory: 12,040 members, including 440 videos totaling 3,126,493,034 bytes, 20 per scene across 22 scenes.

The videos are side-by-side subjective stimuli. At 1600×800, the left 800×800 is reference and the right 800×800 is the degraded NeRF rendering. The paper explicitly describes side-by-side double-stimulus presentation in §II-E: https://arxiv.org/html/2407.08165 . Visual inspection of bowl variants, materials, and 14 scene examples confirms orientation. Full-scene inspection is required before formal annotation.

Model preprocessing crops the right half at original resolution before resizing. Coordinates are recorded against the original encoded frame. No re-rendering, restoration, new view, reference comparison, or generation metadata is provided to models. Source MOS has not been loaded or used.

## Target

The `dyn2024/3DGS-QA` HF snapshot `f35683e629abadfe76a5f75a1a283e67eb2c76b8` lists 3DGS_MOS.csv, metadata, and PLY files; no ready rendered videos. Other distribution endpoints have not been verified. Rendering would change the agreed input/data construction boundary.

The primary target for this 72-hour pilot is therefore **MUGSQA main**, chosen solely for RGB availability before any model metrics. Its published main archive is 13,202,692,013 bytes; extract `.mp4` only and discard PLY streams. Actual revision is recorded in `mugsqa-revision.json`. MOS is saved separately under ignored `data/sealed` and not used for model or protocol selection.

MUGSQA additional remains an optional extension, not a completed second independent benchmark. Source/target grouping, exact duplicates, name overlaps and visual content overlap must be checked. Cross-dataset names alone do not prove content independence. Target human scores may reflect reference-assisted viewing; model inputs remain no-reference.

## Completed inventory and visual audit update

All 440 source / 1,970 target videos downloaded and hashed. `video-inventory.json` records counts. All 22 source scenes now inspected (`source-layout-contact-all.jpg`) and share the verified orientation. Four source groups held out: chair, phonograph, pizza, sofa. Target pizza excluded conservatively (33 videos); 1,937 videos in54 remaining groups.

Detailed multi-view review (`content-overlap-audit.jpg`) shows source vase is a rounded handled blue-white jug, while target baluster vase is slender and lidded. Source piano has a lighter wood/exposed rear structure; target upright piano is reddish ornate wood with a bench and legs. Similar chair/motorcycle/flower subjects also differ in the overview. Reference halves may be inspected by the data curator only for split auditing; algorithm inputs always crop degraded halves. Complete all-target-group overview before finalizing the split.
