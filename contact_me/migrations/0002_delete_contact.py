# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('contact_me', '0001_initial'),
    ]

    operations = [
        migrations.DeleteModel(
            name='Contact',
        ),
    ]
