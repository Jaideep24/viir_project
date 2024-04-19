from operator import mod
from statistics import mode
from xml.parsers.expat import model
from django.db import models

# Create your models here.
class images(models.Model):
    img=models.FileField(blank=True)