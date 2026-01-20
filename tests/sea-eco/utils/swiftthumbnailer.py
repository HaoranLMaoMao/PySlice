import sys,glob,os
import matplotlib.pyplot as plt
sys.path.insert(1,"../../") 

from sea_eco.io import load_swift_to_sea,collect_swift_file,swift_to_sea_metadata
from sea_eco.architecture_numpy.base_structure_numpy import GeneralMetadata

root = sys.argv[-1]

os.makedirs(root+"/thumbnails",exist_ok=True)

#suffix = "h5"
for suffix in ["ndata1","ndata","h5"]:
	# search for files by suffix
	files = glob.glob(root+"/**/*."+suffix,recursive=True)
	for f in files:
		print(f)
		# load signal/metadata
		s = load_swift_to_sea(f)

		# plotting to image file
		dims=None ; ax=None
		# Special cases:
		# 2 spatial dimensions: image or spectrum image
		pos_dims=s.dimensions.position_dimensions
		if len(pos_dims)==2:
			dims=pos_dims
		# Energy axis and spatial dimensions: spectrum image
		#if len(pos_dimes)==2 and "E" in s.dimensions.get_names:
		#	figs,axs=plt.subplots(nrows=2)

		# place images in thumbnails folder
		fo=root+"/thumbnails/"+f.split("/")[-1] # insert "thumbnails" in folder path
		fo=".".join(fo.split(".")[:-1]) # remove suffix

		# large scans are saves as images. poor resolution saves as a matplotlib plot
		if len(pos_dims)==2 and s.data.shape[1]>300:
			s.image(dims=dims,filename=fo) #; plt.clf()
		else:
			s.show(dims=dims,filename=fo) ; plt.clf()



