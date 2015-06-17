from django.conf.urls import url

from . import views

urlpatterns = [
    # ex: /polls/
    url(r'^$', views.contact_page.as_view(), name='contact'),
    # ex: /polls/5/
    url(r'thanks/$', views.thanks, name='thanks'),
]
