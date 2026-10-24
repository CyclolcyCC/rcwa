#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:59:22 2019

@author: kostya
"""

import geometry_2 as geo
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as sg
from descartes.patch import PolygonPatch
from collections.abc import Iterable

def from_points( points ):
    ext =  [(x,y) for x,y in zip(points[::2], points[1::2]) ]
    polygon = sg.Polygon(ext)
    return polygon

def plot_polygon( ax,polygon, color = "#6699cc" ):
    patch = PolygonPatch(polygon, 
                     facecolor= color,
                     lw = 1,
                     edgecolor='k', 
                     alpha=0.4, 
                     zorder=2)
    ax.add_patch(patch)
    ax.plot(*polygon.exterior.xy, lw = 0,color='red', zorder=10)
    return

def get_cursor(pitch,up,dn):
    ext = ( (-pitch/2,dn), 
            (+pitch/2,dn), 
            (+pitch/2,up), 
            (-pitch/2,up))
    return sg.polygon.Polygon(ext)

def get_box(a,b,up,dn):
    ext = (  (a, dn), 
             (b, dn), 
             (b, up), 
             (a, up),  )
        
    return sg.polygon.Polygon(ext)    

def get_area(a,b):
    parts = a.intersection(b)
    if not isinstance(parts,Iterable):
        parts = (parts,)
    
    res = 0.0
    for part in parts:
        res += part.area
    return res

def get_discretization(keys):
    #Here I use the geometry file
    sub, line, ox, cd, interlayer = geo.geometry(keys['cd'],keys['R'],keys['thickness_oxid'],keys['swa'],keys['pitch'],keys['height'],keys['height_offset_substrate'],keys['thickness_oxid_Si_Si3N4'],keys['height_etch_offset'],keys['thickness_oxid_etch_offset'],keys['height_offset_air'])
   
    #sub, line, ox, cd, interlayer =  geo.geometry(keys)
      
    #% Smooth shapes from points
    line = from_points(line)
    interlayer = from_points(interlayer)
    li = line.union(interlayer)
    
    ox = from_points(ox)
    ox = ox.difference(li)
    
    #% Discretization
    #% Define mesh
    eps = 1e-9
    
    pitch = keys["pitch"]
    R = keys["R"]
    top = np.max(ox.exterior.xy[1])
    top_line = np.max(li.exterior.xy[1])+eps
    corner_line = top_line-R
    bttm = 0 + eps
    bttm_ox = bttm + keys["thickness_oxid_etch_offset"]
    corner_ox = bttm_ox + R
    
    
    discr_p = keys["discretization"]
    
    yy1 = np.linspace(top,top_line,discr_p, endpoint = True)
    yy2 = np.flipud(np.linspace(corner_line,top_line,discr_p,endpoint = False))
    yy3 = np.flipud(np.linspace(corner_ox,corner_line,2*discr_p,endpoint = False))
    yy4 = np.flipud(np.linspace(bttm_ox,corner_ox,discr_p,endpoint = False))
    yy5 = np.flipud(np.linspace(bttm, bttm_ox,1,endpoint = False))
    mesh = [*yy1, *yy2, *yy3, *yy4, *yy5]
    
    OX, LI = [], []
    DELTA, COORDS = [],[]
    #% Rasterize
    for up, dn in zip( mesh[:-1] , mesh[1:] ):
        
        DELTA.append( up-dn )
        
        cursor = get_cursor(pitch,up,dn)
        is_in_ox = cursor.intersects(ox)
        is_in_li = cursor.intersects(li)
    
        if is_in_ox and is_in_li:
            gamma_li = get_area(cursor,li)/cursor.area
            gamma_ox = get_area(cursor,ox)/cursor.area
            xl,xr = -pitch/2*gamma_li, +pitch/2*gamma_li
            xll,xrr = xl-pitch/2*gamma_ox, xr+pitch/2*gamma_ox
            part = get_box(xl,xr,up,dn)
            LI.append(part)
            part = get_box(xll,xl,up,dn)
            OX.append(part)
            part = get_box(xr,xrr,up,dn)
            OX.append(part)
            COORDS.append( (xll,xl,xr,xrr) )
            
        elif is_in_ox and not is_in_li:
            gamma_ox = get_area(cursor,ox)/cursor.area    
            xl,xr = -pitch/2*gamma_ox, +pitch/2*gamma_ox
            part = get_box(xl,xr,up,dn)
            OX.append(part)
            COORDS.append( (xl,xr) )
        else:
            raise Exception("Discretization error")

    return DELTA, COORDS, OX, LI





















