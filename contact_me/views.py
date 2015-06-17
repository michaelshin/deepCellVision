from django.shortcuts import render, get_object_or_404
from django.http  import HttpResponse
from .forms import ContactForm
from django.views import generic
from .models import Contact
# Create your views heres

class contact_page(generic.FormView):
    template_name = 'contact_me/contact.html'
    form_class = ContactForm
    success_url = 'thanks/'

    def form_valid(self, form):
        #This method is valid when the form data is POSTed
        #Returns an HttpResponse
        form.send_email()
        return super(contact_page, self).form_valid(form)

def thanks(request):
    return render(request, 'contact_me/thanks.html')
