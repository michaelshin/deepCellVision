from django.shortcuts import render
from django.http  import HttpResponse, HttpResponse
from .forms import CellVisionForm
from . import image_handler
from .models import CellImage
# Create your views here.

def classify(request):
    if request.method == 'POST':
        form = CellVisionForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            choices = form.cleaned_data['options']
            return HttpResponse('image upload success')
    else:
        form = CellVisionForm()
	context_data = {'form': form}
    return render(request, 'cellVision/classify.html', {'form': form})

def segment(request):
    if request.method == 'POST':
        form = CellVisionForm(request.POST, request.FILES)
        if form.is_valid():
            image = CellImage(image = request.FILES['image'])
            image.save()
            path = image.image.path #absolute path of image
            url = image.image.url #relative path of image
            image_handler.show_segment(request.FILES['image'])
            choices = form.cleaned_data['options'] #choices in list form
            return HttpResponse('image upload success')
    else:
        form = CellVisionForm()
	context_data = {'form': form}
    return render(request, "cellVision/segment.html", {'form': form})
