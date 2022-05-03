

## libs and params
library(geosphere) # e.g. distm(c(lon1, lat1), c(lon2, lat2), fun = distHaversine) [I confirmed longitude first]
precision = 0.1 # 0.00001 takes 60s and gets within 0.5meters

## read in data from command line
args = commandArgs(trailingOnly=TRUE)
coords0 = read.table(args[1])
lat = coords0[,1]
long = as.numeric(coords0[,2])
coords = cbind(long,lat) # geosphere wants long first
min_lat = min(lat)
max_lat = max(lat)
min_long = min(long)
max_long = max(long)
mid_long = ((max_long-min_long)/2)+ min_long
mid_lat = ((max_lat-min_lat)/2)+ min_lat
midpoint = c(mid_long, mid_lat)
#print( c(min_lat,max_lat,min_long,max_long ) )

## quick check to make sure the samples don't span over 180 degress
if ( abs(max_lat-min_lat) > 180 ) 
{
	print("samples coords span over 180 degrees lat; the code isn't ready to deal with that")
	quit(save ="no")
}
if ( abs(max_long-min_long) > 180 )
{
        print("samples coords span over 180 degrees long; the code isn't ready to deal with that")
        quit(save ="no")
}

## find a good S—the width of the sampling window— using rough procedure
# fix long, dist bt lats
d1=distm(c(min_long, min_lat), c(min_long, max_lat), fun = distVincentyEllipsoid) # Vincenty = distGeo
# fix other long, dist bt lats
d2=distm(c(max_long, min_lat), c(max_long, max_lat), fun = distVincentyEllipsoid) 
# fix lat, dist bt longs
d3=distm(c(min_long, min_lat), c(max_long, min_lat), fun = distVincentyEllipsoid)
# fix other lat, dist bt longs
d4=distm(c(min_long, max_lat), c(max_long, max_lat), fun = distVincentyEllipsoid)
S = max(d1,d2,d3,d4) 

## bottom left corner of sampling window: min_lat, min_long
corner_bl = c(min_long, min_lat)

## bottom right corner: draw line S distance, same lat.
corner_br = corner_bl # starting on top of the bottom left point
while (distm(corner_bl, corner_br, fun = distVincentyEllipsoid) < S){
	corner_br[1] = corner_br[1] + precision
	}

## top corners: I think I want to find these two corners simultaneously,
b=0
corner_tl = destPoint(corner_bl, b, S)
corner_tr = destPoint(corner_br, -b, S)
while (distm(corner_tl, corner_tr, fun = distVincentyEllipsoid) < S){
	b = b + precision
	corner_tl = destPoint(corner_bl, -b, S)
	corner_tr = destPoint(corner_br, b, S)
	}

## and now get the individual locs.
from_left = dist2Line(coords, rbind(corner_bl, corner_tl))[,1]
from_right = dist2Line(coords, rbind(corner_br, corner_tr))[,1]
total_x = from_left + from_right
final_x = (from_left / total_x)  * S 
#
from_bottom = dist2Line(coords, rbind(corner_bl, corner_br))[,1]
from_top = dist2Line(coords, rbind(corner_tl, corner_tr))[,1]
total_y = from_bottom + from_top
final_y = (from_bottom / total_y)  * S 
#
projection = cbind(final_x,final_y)
projection = projection / 1000 # m to km

## write
write.table(projection, paste(args[1],"_proj",sep=""), sep = "\t", quote = F, row.names=F, col.names=F)

