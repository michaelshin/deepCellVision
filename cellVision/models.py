from django.db import models

# Create your models here.

class CellImage(models.Model):
#    options = 
    image = models.ImageField(upload_to = 'cell_image/%Y/%m/%d')

