#! bash
source activate UniKP

run_dir=$(readlink -f $(dirname $0))

echo Retraining Kcat ...
python ${run_dir}/UniKP_kcat.py
python ${run_dir}/UniKP_kcat_model.py

echo Retraining Km ...
python ${run_dir}/UniKP_Km.py
python ${run_dir}/UniKP_Km_model.py

echo Retraining Kcat/Km ...
python ${run_dir}/UniKP_kcat_Km.py
python ${run_dir}/UniKP_kcat_Km_model.py

echo Done