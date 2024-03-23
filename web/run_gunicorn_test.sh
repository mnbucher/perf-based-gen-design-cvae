cd ..

mkdir -p web/log

> web/log/log_gen_webserver.log
.venv/bin/gunicorn 'wsgi:get_middleware()'