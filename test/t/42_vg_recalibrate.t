#!/usr/bin/env bash

BASH_TAP_ROOT=../deps/bash-tap
. ../deps/bash-tap/bash-tap-bootstrap

PATH=../bin:$PATH # for vg

export LC_ALL="C" # force a consistent sort order 

plan tests 10

vg construct -r small/x.fa -v small/x.vcf.gz >x.vg
vg index -x x.xg -g x.gcsa -k 11 x.vg
vg sim -x x.xg -n 100 -l 100 -i 0.001 --sub-rate 0.01 --random-seed 1 -a > sim.gam
vg map -x x.xg -g x.gcsa -G sim.gam > mapped.gam


vg gamcompare -r 100 mapped.gam sim.gam | vg recalibrate --model recal.model --train -
is $? 0 "vg recalibrate mapping quality model training complete successfully"

vg gamcompare -r 100 mapped.gam sim.gam | vg recalibrate -b --model recal.modelb --train -
is $? 0 "vg recalibrate bag of words model training complete successfully"

vg gamcompare -r 100 mapped.gam sim.gam | vg recalibrate -b -e --model recal.modelbe --train -
is $? 0 "vg recalibrate mems model training complete successfully"

vg gamcompare -r 100 mapped.gam sim.gam | vg recalibrate -e --model recal.modele --train -
is $? 0 "vg recalibrate bag of words and mems model training complete successfully"

vg gamcompare -r 100 mapped.gam sim.gam | vg recalibrate -s --model recal.models --train -
is $? 0 "vg recalibrate mems stats model training complete successfully"

vg recalibrate --model recal.model mapped.gam > /dev/null
is $? 0 "vg recalibrate mapping quality model prediction complete successfully"

vg recalibrate -b --model recal.modelb mapped.gam  > /dev/null
is $? 0 "vg recalibrate bag of words model prediction complete successfully"

vg recalibrate -e --model recal.modele mapped.gam  > /dev/null
is $? 0 "vg recalibrate mems model prediction complete successfully"

vg recalibrate -b -e --model recal.modelbe mapped.gam  > /dev/null
is $? 0 "vg recalibrate bag of words and mems model prediction complete successfully"

vg recalibrate -s --model recal.modelbe mapped.gam  > /dev/null
is $? 0 "vg recalibrate mems stats model prediction complete successfully"


rm -f recal.model* *.gam x.*

