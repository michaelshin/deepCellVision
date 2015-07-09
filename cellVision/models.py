from django.db import models
import os
from time import strftime
# Wrapper to change file names
def update_filename(instance, filename):
    format = strftime('%Y%m%d_%H%M%S') + '.' + instance.image.name.split('.')[-1]
    return format
# Create your models here.

class CellImage(models.Model):
#    options = 
    image = models.ImageField(upload_to = update_filename)

