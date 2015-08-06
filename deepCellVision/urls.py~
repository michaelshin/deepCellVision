""" Default urlconf for deepCellVision """

from django.conf.urls import include, patterns, url
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'deepCellVision.views.home', name='home'),
    url(r'^admin/', include(admin.site.urls)),
    url(r'', include('base.urls', namespace = 'base')),
    url(r'^contact/', include('contact_me.urls', namespace = 'contact_me')),
    url(r'^cellVision/', include('cellVision.urls', namespace = 'cellVision')),
) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
