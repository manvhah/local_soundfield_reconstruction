all: env data figs

data:
	@echo " > downloading data and generating figure folders"
ifneq ("$(wildcard ./data/classroom_frequency_responses.h5)","")
	echo "./data/classroom_frequency_responses.h5 exists already."
else
	mkdir -p data
	cd data
	@echo " > Downloading data, approx 364 MB .."
	wget --no-check-certificate -O ./data/classroom_frequency_responses.h5 https://data.dtu.dk/ndownloader/files/27505451
	wget --no-check-certificate -O ./data/lab_frequency_responses.h5 https://data.dtu.dk/ndownloader/files/27506717
	cd ..
endif

env:
	@echo " > generate and activate python environment"
	python3 -m venv recenv
	@echo " > install sound field reconstruction package and dependencies"
	. ./recenv/bin/activate; cd sfr ; pip install -r requirements.txt ; pip install -e .
	@echo " > ... done"

figs: 
	@echo " > generating DL paper figures"
	mkdir -p figures/decompositions
	mkdir -p figures/reconstructions
	. ./recenv/bin/activate; python3 gen_figures.py
	@echo " > ... done"
