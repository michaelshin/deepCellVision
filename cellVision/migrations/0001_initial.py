# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import cellVision.models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CellImage',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('image', models.ImageField(upload_to=cellVision.models.update_filename)),
                ('frames', models.IntegerField(default=0, null=True)),
                ('channels', models.IntegerField(default=0, null=True)),
                ('target', models.IntegerField(default=0)),
                ('activations', cellVision.models.ListField(default=[])),
                ('name', models.CharField(default=b'', max_length=15)),
            ],
        ),
    ]
