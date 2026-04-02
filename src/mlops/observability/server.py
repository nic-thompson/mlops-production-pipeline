from prometheus_client import start_http_server
import logging

logger = logging.getLogger(__name__)

def start_metrics_server(port: int = 8000):
    """
    Start Prometheus metrics HTTP server.

    Exposes metrics at:
        http://localhost:<port>/metrics
    """
    start_http_server(port)
    logger.info("Prometheus metrics server started on port %s", port)
