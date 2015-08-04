from django.shortcuts import render
from django.http  import HttpResponse, HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.core.files import File 
from django.conf import settings

from .forms import CellVisionForm
import image_handler
from .models import CellImage
import json
from . import classification
# Create your views here.


def classify(request):
    if request.method == 'POST':
        form = CellVisionForm(request.POST, request.FILES)
        if form.is_valid():
            upload = CellImage(image = request.FILES['image'])
            upload.save()
            path = upload.image.path #absolute path of image
            name = upload.image.name #name of the image
            url = upload.image.url
            frames = form.cleaned_data['frames']
            channels = form.cleaned_data['channels']
            target = form.cleaned_data['target']
            choices = form.cleaned_data['options'] #choices in list form
            areas = classification._classify(path, name, frames, channels, target, choices)
            f = [['Class', 'Area']]
            for choice in choices:
                f.append([str(choice), areas[choice]])
            return render(request, 'cellVision/classify_result.html', {'media_url': settings.MEDIA_URL, 'file_name': name, 'url': url, 'activations': json.dumps(f)})
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
            name = upload.image.name #name of the image
            image_handler.show_segment(path, name)
            choices = form.cleaned_data['options'] #choices in list form
            return render(request, 'cellVision/segment_result.html', {'media_url': settings.MEDIA_URL, 'file_name': name.split('.')[0]})
    else:
        form = CellVisionForm()
    return render(request, "cellVision/segment.html", {'form': form})

def download(request, file_name):
    file_location = str(settings.MEDIA_ROOT + "/segment/" + file_name)
    array = File(open(file_location))
    from mimetypes import guess_type
    mime_type, encoding = guess_type(file_name)
    response = HttpResponse(array, content_type=mime_type)
    response['Content-Disposition'] = 'attachment; filename="%s"' %file_name
    return response

