# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import cellVision.models


class Migration(migrations.Migration):

    dependencies = [
        ('cellVision', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cellimage',
            name='image',
            field=models.ImageField(upload_to=cellVision.models.update_filename),
        ),
    ]
