sudo systemctl stop gunicorn

chown -R www-data:www-data /home/git/eth-master-thesis/ckpt

mkdir -p /home/git/eth-master-thesis/web/log

touch /home/git/eth-master-thesis/web/log/log_gen_webserver.log
> /home/git/eth-master-thesis/web/log/log_gen_webserver.log
chown www-data:www-data /home/git/eth-master-thesis/web/log/log_gen_webserver.log

touch /home/git/eth-master-thesis/web/log/log_gunicorn_error.log
> /home/git/eth-master-thesis/web/log/log_gunicorn_error.log
chown www-data:www-data /home/git/eth-master-thesis/web/log/log_gunicorn_error.log

sudo systemctl daemon-reload
sudo systemctl start gunicorn

#sudo systemctl status gunicorn
#watch -n 1 'sudo systemctl status gunicorn'
watch -n 1 'cat /home/git/eth-master-thesis/web/log/log_gen_webserver.log'