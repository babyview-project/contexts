#!/usr/bin/env bash
# Start/stop the annotation site and its MongoDB in tmux sessions (keeps running after you log out).
#
#   scripts/server_ctl.sh start | stop | restart | status | logs
#
# Uses the conda env in ./env if present (Node.js + MongoDB), the port in .env (PORT, default 3000),
# and a MongoDB that only listens on 127.0.0.1 at MONGO_PORT (default 27117) with data in ./db.

set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR=$PWD
[ -d env/bin ] && export PATH="$APP_DIR/env/bin:$PATH"
[ -f .env ] && set -a && . ./.env && set +a

MONGO_PORT=${MONGO_PORT:-27117}
DB_SESSION=annot-db
APP_SESSION=annot-app

running() { tmux has-session -t "$1" 2>/dev/null; }

start() {
  mkdir -p db logs
  if running $DB_SESSION; then echo "MongoDB already running"; else
    tmux new-session -d -s $DB_SESSION \
      "mongod --dbpath '$APP_DIR/db' --bind_ip 127.0.0.1 --port $MONGO_PORT 2>&1 | tee -a '$APP_DIR/logs/mongod.log'"
    echo "Started MongoDB on 127.0.0.1:$MONGO_PORT (tmux: $DB_SESSION)"
    sleep 3
  fi
  if running $APP_SESSION; then echo "Site already running"; else
    tmux new-session -d -s $APP_SESSION "cd '$APP_DIR' && node server.js ${SERVER_FLAGS:-} 2>&1 | tee -a '$APP_DIR/logs/server.log'"
    echo "Started site on port ${PORT:-3000} (tmux: $APP_SESSION)"
  fi
}

stop() {
  running $APP_SESSION && tmux kill-session -t $APP_SESSION && echo "Stopped site"
  # Shut MongoDB down cleanly rather than killing it
  if running $DB_SESSION; then
    mongod --dbpath "$APP_DIR/db" --shutdown >/dev/null 2>&1 || true
    sleep 2
    running $DB_SESSION && tmux kill-session -t $DB_SESSION
    echo "Stopped MongoDB"
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status)
    for s in $DB_SESSION $APP_SESSION; do running $s && echo "$s: running" || echo "$s: stopped"; done
    curl -s -o /dev/null -w "site http://$(hostname):${PORT:-3000}/ -> HTTP %{http_code}\n" "http://127.0.0.1:${PORT:-3000}/experiment/activities.html" || true ;;
  logs) tail -n 40 logs/server.log ;;
  *) echo "Usage: $0 start|stop|restart|status|logs"; exit 1 ;;
esac
