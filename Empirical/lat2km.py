


from geopy import distance
import sys
import numpy as np

# params
precision = 0.001 # 0.00001 takes 30s and gets within 0.5meters

# read data
coords = []
with open(sys.argv[1]) as infile:
    for line in infile:
        newline = list(map(float,line.strip().split()))
        coords.append(newline)
coords = np.array(coords) 
coords[:,0] *= -1 # for testing southern hemisphere
print(coords)
print()

# find max lat and long
min_lat = min(coords[:,0])
max_lat = max(coords[:,0])
min_long = min(coords[:,1])
max_long = max(coords[:,1])
print( min_lat,max_lat,min_long,max_long )
print()

# quick check to make sure the samples don't span over 180 degress
if abs(max_lat-min_lat) > 180 or abs(max_long-min_long) > 180:
    print("samples coords span over 180 degrees lat or long; the code isn't ready to deal with that")
    exit()

# find a good S— that is, the width of the sampling window
lat1 = distance.distance([min_lat,min_long], [max_lat,min_long]).km # confirmed ellipsoid='WGS-84' by default
lat2 = distance.distance([min_lat,max_long], [max_lat,max_long]).km
long1 = distance.distance([min_lat,min_long], [min_lat,max_long]).km
long2 = distance.distance([max_lat,min_long], [max_lat,max_long]).km
S = max([lat1,lat2,long1,long2])
print(S)
print()

# bottom left corner of sampling window
corner_bl = [min_lat, min_long]
print("bottom left", corner_bl)
print()

# bottom right corner: draw line S distance, same lat
corner_br = list(corner_bl) # starting on top of the bottom left point
dist_bottom = 0
while dist_bottom < S:
    corner_br[1] += precision
    dist_bottom = distance.distance(corner_bl, corner_br).km
print("bottom right", corner_br)
print()

# top corners
b=0
corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=0))[0:2] # third val is altitude 
corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=0))[0:2]
dist_top = distance.distance(corner_tl, corner_tr).km
#print(corner_tl,corner_tr,S,dist,S-dist)                                                                                                                     
if (dist_bottom - dist_top) > 0: # e.g. northern hemisphere
    while dist_top < S:
        b += precision
        corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=-b))[0:2]
        corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=b))[0:2]
        dist_top = distance.distance(corner_tl, corner_tr).km
#        print(corner_tl,corner_tr,S,dist_top,S-dist_top)
else: # e.g. southern hemisphere
    while dist_top > S:
        b += precision
        corner_tl = list(distance.distance(kilometers=S).destination(corner_bl, bearing=b))[0:2]
        corner_tr = list(distance.distance(kilometers=S).destination(corner_br, bearing=-b))[0:2]
        dist_top = distance.distance(corner_tl, corner_tr).km
 #       print(corner_tl,corner_tr,S,dist_top,S-dist_top)
print("top left:", corner_tl)
print("top_right:", corner_tr)


# deal with southern hemisphere
# sufficient to just *-1 the latitudes?
# and what if it's on the equator?
# I think the thing to do here is check if dist tl-tr > or < dist bl-br, and flip the bearing symbol accordingly

