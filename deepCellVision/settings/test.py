"""
This is an example settings/test.py file.
Use this settings file when running tests.
These settings overrides what's in settings/base.py
"""

from .base import *


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
        "USER": "",
        "PASSWORD": "",
        "HOST": "",
        "PORT": "",
    },
}

SECRET_KEY = '1ifmp#foi%cnty9by#+#n98l=)5qklvcj5-66$1m(^=ts%d$3o'
