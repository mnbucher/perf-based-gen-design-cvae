cd ..

# sudo yum install gcc openssl-devel bzip2-devel libffi-devel zlib-devel
# sudo yum install -y python3

python3 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="./:$PYTHONPATH"

pip install --upgrade git+https://github.com/VincentStimper/normalizing-flows.git
pip install -r requirements.txt

pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cpu.html
pip install torch-geometric

export FLASK_APP=web/gen_webserver

#export FLASK_ENV=development
export FLASK_ENV=production

#flask run --host=0.0.0.0 --port=5001
#flask run --host=127.0.0.1 --port=5001


# ownership of entire ckpt/ folder to www-data

# ln -s -f /home/git/eth-master-thesis/web/gunicorn.service /etc/systemd/system/gunicorn.service

# sudo systemctl stop gunicorn
# > gunicorn_error.log
# chown www-data:www-data gunicorn_error.log
# sudo systemctl daemon-reload
# sudo systemctl start gunicorn
# sudo systemctl status gunicorn