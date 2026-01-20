# Colors and Color Wheels

SEA-eco bundles custom colormaps and 2D color-wheel utilities tailored for
signal visualization.

## 1D colormaps (`pySEA.sea_eco._plotting.colors`)

- Black-to-color ramps: `cyan_k`, `mage_k`, `yell_k`, `red_k`, `lime_k`,
  `blue_k`.
- Transparent-to-color ramps: `cyan_0`, `mage_0`, `yell_0`, `red_0`, `lime_0`,
  `blue_0`.
- Divergent maps: `cbkrm`, `cbkry` (`arctic_sun` alias), `ckr`, `bkr`, `vkg`,
  `gkv`, `bkg`, `mkg`, `krm`, `kry`, `kbc`.

Usage:

```python
import matplotlib.pyplot as plt
from pySEA.sea_eco._plotting import colors

plt.imshow(data, cmap=colors.cbkry)
plt.show()
```

All colormap names are exported via `__all__` for tab-completion.

## 2D color wheels (`pySEA.sea_eco._plotting.colors2D`)

Functions:

- `get_color_wheel` / `plot_color_wheel`: hue-saturation wheel with optional
  rotation and radius masking.
- `get_color_hexagon` / `plot_color_hexagon`: hexagonal cutout with optional
  vertex labels.
- `plot_rgb_traingle`: Maxwell triangle (RGB or CMY) with label support.

Example:

```python
import matplotlib.pyplot as plt
from pySEA.sea_eco._plotting import colors2D

fig, ax = plt.subplots(figsize=(4, 4))
colors2D.plot_color_hexagon(ax, labels=["A", "B", "C"])
ax.set_axis_off()
plt.show()
```

These helpers are useful for visualizing directional data, phase maps, or
compositional ternaries alongside SEA-eco signals.
