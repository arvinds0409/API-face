# image_classifier_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('classifier.urls')),  # Include your app's URL patterns
]
