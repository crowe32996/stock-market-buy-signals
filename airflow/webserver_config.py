from airflow.www.security import AirflowSecurityManager
from flask_appbuilder.security.manager import AUTH_DB

# Use database authentication
AUTH_TYPE = AUTH_DB
AUTH_ROLE_ADMIN = 'Admin'
AUTH_ROLE_PUBLIC = 'Viewer'
SECURITY_MANAGER_CLASS = AirflowSecurityManager
