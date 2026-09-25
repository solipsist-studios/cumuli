<!--
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: Copyright 2026 Solipsist Studios Inc. (https://solipsist.studio)
-->

# Camera Rig Specs

Each file describes where cameras go. See `docs/synthetic-datasets.md` for the
full reference and `scripts/camera_rig_spec.py` for the resolver.

## Lighting Belongs to the Scene

None of these specs light the subject. A rig spec describes cameras, and the
scene was lit deliberately by whoever authored it.

The `lights` block still exists in the spec language, for a scene that arrives
with no lighting at all. Use it carefully: an earlier version of these specs
carried `{"count": 12, "power_w": 150, "radius": {"subject_heights": 2.1}}`,
which deletes the scene's own lights and replaces them with 5.85 times the
irradiance, rendering the subject 2.55 stops hot. Most of that came from
moving the ring closer rather than from the extra wattage, since irradiance
falls as the inverse square.

For reference, the Ariana studio scene is lit by a ring of eight area lights,
measured from the blend:

| Property | Value |
|---|---|
| Count | 8, evenly spaced |
| Type | AREA, rectangular |
| Power | 100 W each, 800 W total |
| Size | 0.20 x 2.00 m (vertical bars) |
| Radius | 6.00 m from the subject |
| Height | 1.63 m |
| Colour | white, 1.0 / 1.0 / 1.0 |
| World background | 0.051 grey at strength 1.0 |

Written as a spec block, that ring is:

```json
"lights": {"count": 8, "power_w": 100, "radius": 6.0, "height": 1.63,
           "bar_width": 0.2, "bar_height": 2.0}
```

## The Shared Eval Ring

Every spec here carries an identical `eval` block: four cameras at 18.5 degrees
offset and 0.85 of subject height. Identical eval views are what make two runs
comparable, and that offset was chosen so the nearest training camera is 10 to
15 degrees away in every rig here, so no rig is flattered by an eval view
sitting next to one of its own cameras.
