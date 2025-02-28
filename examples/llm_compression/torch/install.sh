ENV_NAME="env"

mkdir -p $HOME/MODEL_DIR
rm -rf $ENV_NAME
python3.11 -m venv $ENV_NAME
. $ENV_NAME/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e ../../../