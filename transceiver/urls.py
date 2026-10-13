# transceiver/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/transmit/', views.transmit_and_analyze_api, name='api_transmit'),
    path('api/decode-wav/', views.decode_wav_api, name='api_decode_wav'),
]