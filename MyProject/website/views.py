from django.shortcuts import render

from django.core.files.storage import FileSystemStorage


def index(request):
    context ={'a':1}
    return render(request, 'index.html',context)

def predictImage(request):
    print(request)

    fileobj = request.FILES['filepath']
    fs = FileSystemStorage()
    filePathName = fs.save(fileobj.name, fileobj)

    context = {'filePathName':filePathName}
    return render(request, 'index.html',context)