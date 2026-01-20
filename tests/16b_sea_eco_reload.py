import os,sys
import matplotlib.pyplot as plt

# Why as an os.system command? I want to ensure no imports remain
if not os.path.exists("05_tacaw.sea"):
    os.system("python3 05_tacaw.py")

if not os.path.exists("sea-eco"):
    u,k=open("/home/qwe/.gitp").readlines() # may need git username and git key for private repo
    u=u.strip() ; k=k.strip()
    os.system("git clone https://"+u+":"+k+"@github.com/sea-ecosystem/sea-eco")

sys.path.insert(1,"sea-eco/src")
from pySEA.sea_eco.io import load
from pySEA.sea_eco.architecture.base_structure_numpy import SEAFile

#loaded = SEAFile()
#loaded.from_sea('05_tacaw.sea')
#loaded.show_tree()

loaded = load(file_path='05_tacaw.sea')
loaded.show_tree()

loaded.show(dims=('frequency','kx'))
plt.show()

loaded[:,30:35,:,:].show(dims=('kx','ky'))
plt.show()
