from django.db import models
from django.utils import timezone
import os
import time
import hashlib
import ast

class ListField(models.TextField):
    __metaclass__ = models.SubfieldBase
    description = "Stores a python list"

    def __init__(self, *args, **kwargs):
        super(ListField, self).__init__(*args, **kwargs)

    def to_python(self, value):
        if not value:
            value = []

        if isinstance(value, list):
            return value

        return ast.literal_eval(value)

    def get_prep_value(self, value):
        if value is None:
            return value

        return unicode(value)

    def value_to_string(self, obj):
        value = self._get_val_from_obj(obj)
        return self.get_db_prep_value(value)

# Wrapper to change file names
def update_filename(instance, filename):
    format = time.strftime('%Y%m%d%H%M%S') + '.' + instance.image.name.split('.')[-1]
    return format

# Create your models here.

class CellImage(models.Model):
    image = models.ImageField(upload_to = update_filename)
    frames = models.IntegerField(default=0, null = True)
    channels = models.IntegerField(default=0, null = True)
    target = models.IntegerField(default=0)
    activations = ListField(default = [])
    name = models.CharField(max_length=15, default= '')
    last_accessed = timezone.now()
    email = models.EmailField(default = '')
    
    def __unicode__(self):
        return self.name

 
