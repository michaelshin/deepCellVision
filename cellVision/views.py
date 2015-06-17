from django.shortcuts import render

# Create your views here.

def classify(request):
    return render(request, "cellVision/classify.html")

def segment(request):
    return render(request, "cellVision/segment.html")
