#! bash
source activate UniKP

echo Retraining Kcat ...
python /repo/UniKP/scripts/UniKP_kcat.py

echo Retraining Km ...
python /repo/UniKP/scripts/UniKP_Km.py

echo Retraining Kcat/Km ...
python /repo/UniKP/scripts/UniKP_kcat_Km.py

echo Done