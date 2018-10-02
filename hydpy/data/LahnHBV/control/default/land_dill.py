# -*- coding: utf-8 -*-

from hydpy.models.hland_v1 import *

simulationstep("1h")
parameterstep("1d")

area(692.3)
nmbzones(12)
zonetype(FIELD, FOREST, FIELD, FOREST, FIELD, FOREST, FIELD, FOREST, FIELD,
         FOREST, FIELD, FOREST)
zonearea(14.41, 7.0599999999999996, 70.829999999999998, 84.359999999999999,
         70.969999999999999, 198.0, 27.75, 130.0, 27.280000000000001,
         56.939999999999998, 1.0900000000000001, 3.6099999999999999)
zonez(2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0)
zrelp(3.75)
zrelt(3.75)
zrele(3.665)
pcorr(1.0)
pcalt(0.10000000000000001)
rfcf(1.0428299999999999)
sfcf(1.1000000000000001)
tcalt(0.59999999999999998)
ecorr(1.0)
ecalt(0.0)
epf(0.02)
etf(0.10000000000000001)
ered(0.0)
ttice(nan)
icmax(field=1.0, forest=1.5)
tt(0.55823999999999996)
ttint(2.0)
dttm(0.0)
cfmax(field=4.5585300000000002, forest=2.7351179999999999)
gmelt(0.0)
cfr(0.050000000000000003)
whc(0.10000000000000001)
fc(278.0)
lp(0.90000000000000002)
beta(2.5401099999999999)
percmax(1.39636)
cflux(0.0)
resparea(True)
recstep(1200.0)
alpha(1.0)
k(0.005617743528874685)
k4(0.05646)
gamma(0.0)
maxbaz(0.36728000000000005)
abstr(0.0)
