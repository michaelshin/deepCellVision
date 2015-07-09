from django.shortcuts import render
from django.http  import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.files import File 
from django.conf import settings

from .forms import CellVisionForm
from .image_handler import show_segment
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
            name = request.FILES['image'].name.split('.')[0] #name of the image
            segmented = show_segment(path, name)
            choices = form.cleaned_data['options'] #choices in list form
            return render(request, 'cellVision/segment_result.html', {'media_url': settings.MEDIA_URL, 'segmented': segmented, 'file_name': name})
    else:
        form = CellVisionForm()
	context_data = {'form': form}
    return render(request, "cellVision/segment.html", {'form': form})

def download(request):
    file_location = "/home/michael/deepCellVision/media/segment/2015/07/08/001003000.npy"
    array = File(open(file_location))
    response = HttpResponse(array, content_type='application/octet-stream')
    response['Content-Disposition'] = 'attachment; filename="array.npy"'
    return response

