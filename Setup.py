#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:39:53 2023

@author: J. Lengyel, January 2023, janka.lengyel@ens-lyon.fr
"""

from setuptools import setup

setup(
   name='MASPP',
   version='0.1.0',
   author=['Stephane G. Roux, Janka Lengyel'],
   author_email='janka.lengyel@ens-lyon.fr',
   scripts=['scripts/MASPP.py'],
   #url='',
   license='LICENSE_MASPP.txt',
   description='Multiscale Analysis of Spatial Point Processes',
   #long_description=open('README.txt').read(),
   #install_requires=[],
)