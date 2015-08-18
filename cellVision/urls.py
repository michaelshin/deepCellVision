from django.conf.urls import url

from . import views

urlpatterns = [
   url(r'classify/$', views.classify, name='classify'),
   url(r'media/(?P<file_name>\d+)/$', views.download),
   url(r'results/(?P<file_name>\d+)/$', views.results, name ='results'),
   url(r'sample/(?P<num>\d+)/$', views.sample),
]
