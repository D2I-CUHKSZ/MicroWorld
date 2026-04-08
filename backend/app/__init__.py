
import os
import warnings


warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .setting.settings import Config
from .infrastructure.logger import setup_logger, get_logger


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)


    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False


    logger = setup_logger('lightworld')


    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process

    if should_log_startup:
        logger.info("=" * 50)
        logger.info("LightWorld Backend 启动中...")
        logger.info("=" * 50)


    CORS(app, resources={r"/api/*": {"origins": "*"}})


    from .utils.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("已注册模拟进程清理函数")


    @app.before_request
    def log_request():
        logger = get_logger('lightworld.request')
        logger.debug(f"请求: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"请求体: {request.get_json(silent=True)}")

    @app.after_request
    def log_response(response):
        logger = get_logger('lightworld.request')
        logger.debug(f"响应: {response.status_code}")
        return response


    from .adapters.http import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')


    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'LightWorld Backend'}

    if should_log_startup:
        logger.info("LightWorld Backend 启动完成")

    return app
