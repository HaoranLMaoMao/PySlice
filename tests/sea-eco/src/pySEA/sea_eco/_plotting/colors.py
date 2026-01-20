"""Custom matplotlib colormaps for SEA-eco plotting."""

from matplotlib.colors import LinearSegmentedColormap


##################
# Black to color #
##################
cyan_k = [(0, 0, 0, 1), (0, 1, 1, 1)]
cyan_k = LinearSegmentedColormap.from_list("cyan_k", cyan_k, N=255)

mage_k = [(0, 0, 0, 1), (1, 0, 1, 1)]
mage_k = LinearSegmentedColormap.from_list("mage_k", mage_k, N=255)

yell_k = [(0, 0, 0, 1), (1, 1, 0, 1)]
yell_k = LinearSegmentedColormap.from_list("yell_k", yell_k, N=255)

red_k = [(0, 0, 0, 1), (1, 0, 0, 1)]
red_k = LinearSegmentedColormap.from_list("red_k", red_k, N=255)

lime_k = [(0, 0, 0, 1), (0, 1, 0, 1)]
lime_k = LinearSegmentedColormap.from_list("lime_k", lime_k, N=255)

blue_k = [(0, 0, 0, 1), (0, 0, 1, 1)]
blue_k = LinearSegmentedColormap.from_list("blue_k", blue_k, N=255)


##################
# Clear to color #
##################
cyan_0 = [(0, 0, 0, 0), (0, 1, 1, 1)]
cyan_0 = LinearSegmentedColormap.from_list("cyan_0", cyan_0, N=255)

mage_0 = [(0, 0, 0, 0), (1, 0, 1, 1)]
mage_0 = LinearSegmentedColormap.from_list("mage_0", mage_0, N=255)

yell_0 = [(0, 0, 0, 0), (1, 1, 0, 1)]
yell_0 = LinearSegmentedColormap.from_list("yell_0", yell_0, N=255)

red_0 = [(0, 0, 0, 0), (1, 0, 0, 1)]
red_0 = LinearSegmentedColormap.from_list("red_0", red_0, N=255)

lime_0 = [(0, 0, 0, 0), (0, 1, 0, 1)]
lime_0 = LinearSegmentedColormap.from_list("lime_0", lime_0, N=255)

blue_0 = [(0, 0, 0, 0), (0, 0, 1, 1)]
blue_0 = LinearSegmentedColormap.from_list("blue_0", blue_0, N=255)

##################
#    Divergent   #
##################
cbkrm = [(0,1,1),(0,0,1),(0,0,0),(1,0,0),(1,0,1)]
cbkrm = LinearSegmentedColormap.from_list("cbkrm", cbkrm, N=255)

cbkry = [(0,1,1), (0,0,1), (0,0,0), (1,0,0), (1,1,0)]
cbkry = LinearSegmentedColormap.from_list("arctic_sun ", cbkry , N=255)
arctic_sun = cbkry

ckr = [(0,1,1), (0,0,0), (1,0,0)]
ckr = LinearSegmentedColormap.from_list("ckr", ckr, N=255)

bkr = [[0,0,1],[0,0,0],[1,0,0]]
bkr = LinearSegmentedColormap.from_list("bkr", bkr, N=255)

vkg = [[0.5,0,1], [0,0,0], [0.5,1,0]]
vkg = LinearSegmentedColormap.from_list("vkg", vkg, N=255)

gkv = [[0.5,1,0], [0,0,0], [0.5,0,1]]
gkv = LinearSegmentedColormap.from_list("gkv", gkv, N=255)

bkg = [[0,0,1],[0,0,0],[0,1,0]]
bkg = LinearSegmentedColormap.from_list("bkg", bkg, N=255)

mkg = [[1,0,1],[0,0,0],[0,1,0]]
mkg = LinearSegmentedColormap.from_list("mkg", mkg, N=255)

krm = [[1,0,1],[1,0,0],[0,0,0]][::-1]
krm = LinearSegmentedColormap.from_list("krm", krm, N=255)

kry = [[1,1,0],[1,0,0],[0,0,0]][::-1]
kry = LinearSegmentedColormap.from_list("kry", kry, N=255)

kbc = [[0,0,0],[0,0,1],[0,1,1]]
kbc = LinearSegmentedColormap.from_list("kbc", kbc, N=255)


#Direct imports to only the cmaps
__cmaps = [k for k,v in locals().items() if isinstance(v, LinearSegmentedColormap)]
__all__ = __cmaps

def __dir__():
    return sorted(__all__)
