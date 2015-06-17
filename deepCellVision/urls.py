""" Default urlconf for deepCellVision """

from django.conf.urls import include, patterns, url
from django.contrib import admin
admin.autodiscover()


def bad(request):
    """ Simulates a server error """
    1 / 0

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'deepCellVision.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^bad/$', bad),
    url(r'', include('base.urls', namespace = 'base')),
    url(r'^contact/', include('contact_me.urls', namespace = 'contact_me')),
    url(r'^cellVision/', include('cellVision.urls', namespace = 'cellVision')),
)

