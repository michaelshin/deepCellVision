from django.test import TestCase
from contact_me.models import Contact

# Create your tests here.

class ContactTests(TestCase):
    '''Contact form model tests'''
    
    def test_str(self):
        contact = Contact(first_name = 'John', last_name = 'Smith')
        self.assertEquals(str(contact), 'John Smith')
    
