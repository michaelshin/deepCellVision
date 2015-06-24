from django import forms
#Create your own forms below.

class CellVisionForm(forms.Form):
    #accepts file types only listed from http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    image = forms.ImageField(error_messages={'required': 'Please enter a path location', 'invalid': 'Please enter a valid image file'}, help_text = "Add an image file (BMP, EPS, GIF, IM, JPEG (2000), MSP, PCX, PNG, PPM, SPIDER, TIFF, WebP, XBM, XVThumbnails accepted)")
    options = forms.MultipleChoiceField([('ACTIN','Actin'),('BUDNECK','Budneck'), ('BUDTIP','Budtip'), ('CELLPERIPHERY','Cell Periphery'), ('CYTOPLASM','Cytoplasm'), ('ENDOSOME', 'Endosome'), ('ER', 'ER'), ('GOLGI','Golgi Body'), ('MITOCHONDRIA', 'Mitocondria'), ('NUCLEARPERIPHERY','Nuclear Periphery'), ('NUCLEI','Nuclei'), ('NUCLEOLUS','Nucleolus'), ('PEROXISOME','Peroxisome'), ('SPINDLE','Spindle'), ('SPINDLEPOLE','Spindlepole'), ('VACUOLARMEMBRANE','Vacuolar Membrane'), ('VACUOLE', 'Vacuole')], widget = forms.CheckboxSelectMultiple, help_text = "Choose all subcellular structures and organelles you want to find")

    
