#FSHJAR002 MAKEFILE ML ASS 3

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt
	
venv:
	test -d venv || python3 -m venv venv
	
clean:
	rm -rf venv
	find -iname "*.pyc" -delete

runMain:
	venv/bin/python3 src/wrapper.py "./trained_models/simple_model" -m "simple"

runMainLoad:
	venv/bin/python3 src/wrapper.py "./trained_models/simple_model" -m "simple" -ls -e 15 

runMainSkip:
	venv/bin/python3 src/wrapper.py "./trained_models/simple_model" -m "simple" -s

runLSTM:
	venv/bin/python3 src/wrapper.py "./trained_models/lstm_model" -m "LSTM"

runLSTMLoad:
	venv/bin/python3 src/wrapper.py "./trained_models/lstm_model" -m "LSTM" -ls

runHelp:
	venv/bin/python3 src/wrapper.py -h