import pandas as pd

HOwt_all_cells = pd.read_csv('/home/morphology/mpg3/oren/Data_Sets/Yeast_Protein_Localization/Yolanda_Chong/\
HOwt_all_cell_locs_shape_v1.csv', names = [u'cScreen', u'ImNumber', u'fileNameGfp', u'fileNameRFP', u'fileNameOverlay',
                                           u'nwellid',u'cellLocCent_X',u'cellLocCent_Y', u'cells_AreaShape_Area',
                                           u'cells_AreaShape_Eccentricity', u'cells_AreaShape_EulerNumber',
                                           u'cells_AreaShape_Extent,', u'cells_AreaShape_FormFactor',
                                           u'cells_AreaShape_MajorAxisLength',u'cells_AreaShape_MinorAxisLength',
                                           u'cells_AreaShape_Orientation', u'cells_AreaShape_Perimeter',
                                           u'cells_AreaShape_Solidity'])
imagePath = []
path2Yolanda_images = '/home/morphology/Yolanda/Genome_wide/GFP_Genome_wide/HOwt/'

for image_index in range(0,6000):
    path_to_cur_image = HOwt_all_cells.fileNameGfp[image_index][:-2]+'.flex'
    curImagePath = path2Yolanda_images+path_to_cur_image
    if curImagePath not in imagePath:
        imagePath.append(curImagePath)

print imagePath
f = open('path_to_image.txt', 'w')
for path in imagePath:
	f.write(path + '\n')
f.close()