

#import geopy
from geopy import distance
import sys
import numpy as np

# read data
coords = []
with open(sys.argv[1]) as infile:
    for line in infile:
        newline = list(map(float,line.strip().split()))
        coords.append(newline)
coords = np.array(coords)
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
lat1 = distance.distance((min_long, min_lat), (min_long, max_lat)).km # confirmed ellipsoid='WGS-84' by default
lat2 = distance.distance((max_long, min_lat), (max_long, max_lat)).km
long1 = distance.distance((min_long, min_lat), (max_long, min_lat)).km
long2 = distance.distance((min_long, max_lat), (max_long, max_lat)).km
S = max(lat1,lat2,long1,long2)
print(lat1,lat2,long1,long2,S)






# destination point...
#geopy.distance.distance(miles=10).destination((34, 148), bearing=90)
