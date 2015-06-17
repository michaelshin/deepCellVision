from django import forms

#Create your own forms below.

class ContactForm(forms.Form):
    name = forms.CharField()
    message = forms.CharField(widget = forms.Textarea)

    def send_email(self):
        #send emaili using the self.cleaned_data dict
        pass
