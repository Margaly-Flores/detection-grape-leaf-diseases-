#!/bin/bash
gunicorn wsgi:app --daemon
rq worker --with-scheduler