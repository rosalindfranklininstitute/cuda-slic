

from .base import register_view
from ..components import *


@register_view(name='load_workspace', title='Load Workspace')
class WorkspaceLoader(QCSWidget):
    pass