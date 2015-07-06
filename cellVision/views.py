from django.shortcuts import render
from django.http  import HttpResponse, HttpResponse
from .forms import CellVisionForm
from . image_handler import show_segment
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
            upload = CellImage(image = request.FILES['image'])
            upload.save()
            path = upload.image.path #absolute path of image
            url = upload.image.url #relative path of image
            show_segment(path)
            choices = form.cleaned_data['options'] #choices in list form
            return HttpResponse('image upload success')
    else:
        form = CellVisionForm()
	context_data = {'form': form}
    return render(request, "cellVision/segment.html", {'form': form})
