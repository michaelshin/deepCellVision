"""urlconf for the base application"""

from django.conf.urls import url, patterns
from . import views

urlpatterns = patterns('base.views',
    url(r'^$', 'home', name='home'),
    url(r'^about$', views.about, name ='about'),
)
