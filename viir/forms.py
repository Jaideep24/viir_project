from django import forms
from .models import*
class Imageaform(forms.ModelForm):
    img=forms.FileField(required=True)
    class Meta:
        model=images
        fields=("img",)